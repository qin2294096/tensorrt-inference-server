// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include "src/core/api.pb.h"
#include "src/core/constants.h"
#include "src/core/provider.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "src/core/scheduler_utils.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

using ModelQueuePolicyMap =
    ::google::protobuf::Map<::google::protobuf::uint32, ModelQueuePolicy>;

class RequestQueue {
 public:
  RequestQueue()
      : timeout_action_(ModelQueuePolicy::REJECT),
        default_timeout_microseconds_(0),
        allow_timeout_override_(false),
        max_queue_size_(0)
  {
  }
  
  RequestQueue(const ModelQueuePolicy& policy)
      : timeout_action_(policy.timeout_action()),
        default_timeout_microseconds_(policy.default_timeout_microseconds()),
        allow_timeout_override_(policy.allow_timeout_override()),
        max_queue_size_(policy.max_queue_size())
  {
  }

  template <typename... Args>
  Status Emplace(Args&&... args)
  {
    if ((max_queue_size_ != 0) && (queue_.size() >= max_queue_size_)) {
      return Status(
          RequestStatusCode::UNAVAILABLE, "Exceeds maximum queue size");
    }
    queue_.emplace_back(args...);
    auto timeout_microseconds = default_timeout_microseconds_;
    if (allow_timeout_override_ &&
        queue_.back()
                .request_provider_->RequestHeader()
                .timeout_microseconds() != 0) {
      timeout_microseconds = queue_.back()
                                 .request_provider_->RequestHeader()
                                 .timeout_microseconds();
    }
    if (timeout_microseconds != 0) {
      struct timespec now;
      clock_gettime(CLOCK_MONOTONIC, &now);
      timeout_timestamp_ns_.emplace_back(
          TIMESPEC_TO_NANOS(now) + timeout_microseconds * 1000);
    } else {
      timeout_timestamp_ns_.emplace_back(0);
    }

    return Status::Success;
  }

  Scheduler::Payload Dequeue()
  {
    if (!queue_.empty()) {
      auto res = std::move(queue_.front());
      queue_.pop_front();
      timeout_timestamp_ns_.pop_front();
      return res;
    } else {
      auto res = std::move(delayed_queue_.front());
      delayed_queue_.pop_front();
      return res;
    }
  }

  void ApplyPolicy()
  {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    auto now_nanoseconds = TIMESPEC_TO_NANOS(now);
    for (size_t idx = 0; idx != queue_.size();) {
      if ((timeout_timestamp_ns_[idx] != 0) && (now_nanoseconds > timeout_timestamp_ns_[idx])) {
        if (timeout_action_ == ModelQueuePolicy::DELAY) {
          delayed_queue_.emplace_back(std::move(queue_[idx]));
        } else {
          rejected_queue_.emplace_back(std::move(queue_[idx]));
        }
        queue_.erase(queue_.begin() + idx);
        timeout_timestamp_ns_.erase(timeout_timestamp_ns_.begin() + idx);
      } else {
        // only advance the index if the current item is still valid
        idx++;
      }
    }
  }

  bool ApplyPolicy(size_t idx)
  {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    auto now_nanoseconds = TIMESPEC_TO_NANOS(now);
    while (idx < queue_.size()) {
      if ((timeout_timestamp_ns_[idx] != 0) && (now_nanoseconds > timeout_timestamp_ns_[idx])) {
        if (timeout_action_ == ModelQueuePolicy::DELAY) {
          delayed_queue_.emplace_back(std::move(queue_[idx]));
        } else {
          rejected_queue_.emplace_back(std::move(queue_[idx]));
        }
        queue_.erase(queue_.begin() + idx);
        timeout_timestamp_ns_.erase(timeout_timestamp_ns_.begin() + idx);
      } else {
        // Current idx is pointing to an item with unexpired timeout
        return true;
      }
    }
    // At this point, idx is pointing to an item with expired timeout.
    // If the item is in delayed queue, then return true. Otherwise, false
    // meaning the queue has no item with this 'idx'.
    return ((idx - queue_.size()) < delayed_queue_.size());
  }

  std::deque<Scheduler::Payload> ReleaseRejectedQueue()
  {
    std::deque<Scheduler::Payload> res;
    rejected_queue_.swap(res);
    return res;
  }

  Scheduler::Payload& At(size_t idx)
  {
    if (idx < queue_.size()) {
      return queue_[idx];
    } else {
      return delayed_queue_[idx - queue_.size()];
    }
  }

  uint64_t TimeoutAt(size_t idx)
  {
    if (idx < queue_.size()) {
      return timeout_timestamp_ns_[idx];
    } else {
      return 0;
    }
  }

  bool Empty() { return Size() == 0; }

  size_t Size() { return queue_.size() + delayed_queue_.size(); }

 private:
  std::deque<Scheduler::Payload> queue_;
  std::deque<uint64_t> timeout_timestamp_ns_;
  std::deque<Scheduler::Payload> delayed_queue_;
  std::deque<Scheduler::Payload> rejected_queue_;
  const ModelQueuePolicy::TimeoutAction timeout_action_;
  const uint64_t default_timeout_microseconds_;
  const bool allow_timeout_override_;
  const uint32_t max_queue_size_;
};

class PriorityQueue {
 public:
  PriorityQueue()
  {
    ModelQueuePolicy default_policy;
    queues_.emplace(0, RequestQueue(default_policy));
    ResetCursor();
  }

  PriorityQueue(
      const ModelQueuePolicy& default_queue_policy, uint32_t priority_levels,
      const ModelQueuePolicyMap queue_policy_map)
  {
    if (priority_levels == 0) {
      queues_.emplace(0, RequestQueue(default_queue_policy));
    } else {
      for (uint32_t level = 1; level <= priority_levels; level++) {
        auto it = queue_policy_map.find(level);
        if (it == queue_policy_map.end()) {
          queues_.emplace(level, RequestQueue(default_queue_policy));
        } else {
          queues_.emplace(level, RequestQueue(it->second));
        }
      }
    }
    ResetCursor();
  }

  template <typename... Args>
  Status Emplace(uint32_t priority_level, Args&&... args)
  {
    auto status = queues_[priority_level].Emplace(args...);
    if (status.IsOk()) {
      pending_cursor_.valid_ &=
          (priority_level > pending_cursor_.curr_it_->first);
    }
    return status;
  }

  Scheduler::Payload Dequeue()
  {
    pending_cursor_.valid_ = false;
    for (auto& queue : queues_) {
      if (!queue.second.Empty()) {
        return queue.second.Dequeue();
      }
    }
    throw std::out_of_range("dequeue on empty queue");
  }

  size_t Size()
  {
    size_t res = 0;
    for (auto& queue : queues_) {
      res += queue.second.Size();
    }
    return res;
  }

  bool Empty() { return Size() == 0; }

  Scheduler::Payload& PayloadAtCursor() { return pending_cursor_.GetItem(); }

  void MarkCursor() { current_mark_ = pending_cursor_; }

  void Next() { pending_cursor_.Next(); }

  bool CursorEnd()
  {
    return pending_cursor_.curr_it_ == pending_cursor_.end_it_;
  }

  void ResetCursor()
  {
    pending_cursor_ = Cursor(queues_.begin(), queues_.end());
  }

  void SetCursorToMark() { pending_cursor_ = current_mark_; }

  bool IsCursorValid()
  {
    if (pending_cursor_.valid_) {
      struct timespec now;
      clock_gettime(CLOCK_MONOTONIC, &now);
      return TIMESPEC_TO_NANOS(now) <
             pending_cursor_.pending_batch_closest_timeout_ns_;
    }
    return false;
  }

  uint64_t OldestEnqueueTime()
  {
    return pending_cursor_.pending_batch_oldest_enqueue_time_ns_;
  }

  uint64_t ClosestTimeout()
  {
    return pending_cursor_.pending_batch_closest_timeout_ns_;
  }

 private:
  using PriorityQueues = std::map<uint32_t, RequestQueue>;
  PriorityQueues queues_;

  struct Cursor {
    Cursor() = default;
    Cursor(PriorityQueues::iterator start_it, PriorityQueues::iterator end_it)
        : curr_it_(start_it), end_it_(end_it), queue_idx_(0),
          pending_batch_closest_timeout_ns_(0),
          pending_batch_oldest_enqueue_time_ns_(0), valid_(false)
    {
      while (curr_it_ != end_it_) {
        if (!(curr_it_->second.ApplyPolicy(queue_idx_))) {
          curr_it_++;
          queue_idx_ = 0;
        } else {
          pending_batch_closest_timeout_ns_ =
              curr_it_->second.TimeoutAt(queue_idx_);
          pending_batch_oldest_enqueue_time_ns_ = TIMESPEC_TO_NANOS(
              curr_it_->second.At(queue_idx_).stats_->Timestamp(
                  ModelInferStats::TimestampKind::kQueueStart));
          valid_ = true;
          break;
        }
      }
    }

    Cursor(const Cursor& rhs)
        : curr_it_(rhs.curr_it_), end_it_(rhs.end_it_),
          queue_idx_(rhs.queue_idx_),
          pending_batch_closest_timeout_ns_(
              rhs.pending_batch_closest_timeout_ns_),
          pending_batch_oldest_enqueue_time_ns_(
              rhs.pending_batch_oldest_enqueue_time_ns_),
          valid_(rhs.valid_)
    {
    }

    Scheduler::Payload& GetItem()
    {
      return curr_it_->second.At(queue_idx_);
    }

    void Next()
    {
      const auto& timeout_ns = curr_it_->second.TimeoutAt(queue_idx_);
      if (timeout_ns != 0) {
        if (pending_batch_closest_timeout_ns_ != 0) {
          pending_batch_closest_timeout_ns_ =
              std::min(pending_batch_closest_timeout_ns_, timeout_ns);
        } else {
          pending_batch_closest_timeout_ns_ = timeout_ns;
        }
      }
      pending_batch_oldest_enqueue_time_ns_ = std::min(
          pending_batch_oldest_enqueue_time_ns_,
          TIMESPEC_TO_NANOS(
              curr_it_->second.At(queue_idx_).stats_->Timestamp(
                  ModelInferStats::TimestampKind::kQueueStart)));
      ++queue_idx_;
      while (curr_it_ != end_it_) {
        if (!(curr_it_->second.ApplyPolicy(queue_idx_))) {
          curr_it_++;
          queue_idx_ = 0;
        } else {
          break;
        }
      }
    }

    PriorityQueues::iterator curr_it_;
    PriorityQueues::iterator end_it_;
    size_t queue_idx_;
    uint64_t pending_batch_closest_timeout_ns_;
    uint64_t pending_batch_oldest_enqueue_time_ns_;
    bool valid_;
  };

  Cursor pending_cursor_;
  Cursor current_mark_;
};

// Scheduler that implements dynamic batching.
class DynamicBatchScheduler : public Scheduler {
 public:
  // Create a scheduler to support a given number of runners and a run
  // function to call when a request is scheduled.
  static Status Create(
      const uint32_t runner_id_start, const uint32_t runner_cnt, const int nice,
      const StandardInitFunc& OnInit, const StandardWarmupFunc& OnWarmup,
      const StandardRunFunc& OnSchedule,
      const StandardShapeTensorPeekFunc& OnPeek,
      const bool dynamic_batching_enabled,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const bool preserve_ordering,
      const std::set<int32_t>& preferred_batch_sizes,
      const uint64_t max_queue_delay_microseconds,
      std::unique_ptr<Scheduler>* scheduler);

  // Create a scheduler to support a given number of runners and a run
  // function to call when a request is scheduled. And the scheduler also
  // supports different queue policies for different priority levels.
  static Status Create(
      const uint32_t runner_id_start, const uint32_t runner_cnt, const int nice,
      const StandardInitFunc& OnInit, const StandardWarmupFunc& OnWarmup,
      const StandardRunFunc& OnSchedule,
      const StandardShapeTensorPeekFunc& OnPeek,
      const bool dynamic_batching_enabled,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const bool preserve_ordering,
      const std::set<int32_t>& preferred_batch_sizes,
      const uint64_t max_queue_delay_microseconds,
      const ModelQueuePolicy& default_queue_policy,
      const uint32_t priority_level, const ModelQueuePolicyMap& queue_policy_map,
      std::unique_ptr<Scheduler>* scheduler);

  ~DynamicBatchScheduler();

  // \see Scheduler::Enqueue()
  void Enqueue(
      const std::shared_ptr<ModelInferStats>& stats,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const std::shared_ptr<InferResponseProvider>& response_provider,
      std::function<void(const Status&)> OnComplete) override;

 private:
  DynamicBatchScheduler(
      const uint32_t runner_id_start, const uint32_t runner_cnt,
      const StandardInitFunc& OnInit, const StandardWarmupFunc& OnWarmup,
      const StandardRunFunc& OnSchedule,
      const StandardShapeTensorPeekFunc& OnPeek,
      const bool dynamic_batching_enabled,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const bool preserve_ordering,
      const std::set<int32_t>& preferred_batch_sizes,
      const uint64_t max_queue_delay_microseconds);
  void SchedulerThread(
      const uint32_t runner_id, const int nice,
      const std::shared_ptr<std::atomic<bool>>& rthread_exit,
      std::promise<bool>* is_initialized);
  uint64_t GetDynamicBatch(const int64_t runner_id);
  void FinalizePayloads(
      const uint32_t runner_id,
      std::shared_ptr<std::vector<Scheduler::Payload>> payloads,
      const Status& status);

  // Function the scheduler will call to initialize a runner.
  const StandardInitFunc OnInit_;

  // Function the scheduler will call to warmup a runner.
  const StandardWarmupFunc OnWarmup_;

  // Function the scheduler will call to schedule a payload(s) for
  // execution.
  const StandardRunFunc OnSchedule_;

  // Function the scheduler will call to peek at shape tensors.
  const StandardShapeTensorPeekFunc OnPeek_;

  // True if dynamic batching is enabled.
  const bool dynamic_batching_enabled_;

  // The number of scheduler threads.
  const uint32_t scheduler_thread_cnt_;

  // The number of scheduler threads currently idle.
  uint32_t idle_scheduler_thread_cnt_;

  // Mutex and condvar protecting the scheduling queue.
  std::mutex mu_;
  std::condition_variable cv_;

  // Map from priority level to queue holding inference requests for the model
  // represented by this scheduler. Priority level 0 is reserved for default
  // timeout queue
  PriorityQueue queue_;

  std::vector<std::unique_ptr<std::thread>> scheduler_threads_;
  std::vector<std::shared_ptr<std::atomic<bool>>> scheduler_threads_exit_;

  size_t max_preferred_batch_size_;
  std::set<int32_t> preferred_batch_sizes_;
  uint64_t pending_batch_delay_ns_;
  size_t pending_batch_size_;
  size_t pending_batch_queue_cnt_;
  PendingBatchShapes pending_batch_shapes_;

  size_t queued_batch_size_;
  size_t next_preferred_batch_size_;

  // The input tensors that require shape checking before being
  // allowed in a batch. As a map from the tensor name to a bool. If
  // tensor is in map then its shape must match shape of same tensor
  // in requests already in the batch. If value is "true" then
  // additional tensor is treated as a shape tensor and the values
  // contained in the shape tensor must match same tensor already in
  // the batch.
  const std::unordered_map<std::string, bool> enforce_equal_shape_tensors_;

  // If true the ordering of responses matches the order of requests
  // even when there are multiple scheduler threads.
  const bool preserve_ordering_;

  // Holds the sequence of runner indices in order the payloads were issued.
  std::queue<size_t> runner_queue_;
  // Lock to protect the runner_queue_
  std::mutex runner_queue_mtx_;
  // Per runner queues to store the ready payloads
  std::vector<std::queue<std::shared_ptr<std::vector<Scheduler::Payload>>>>
      completion_queues_;
  // Lock to protect the completion_queues_
  std::mutex completion_queues_mtx_;
};

}}  // namespace nvidia::inferenceserver
