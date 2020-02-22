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
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "src/core/scheduler_utils.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class RequestQueue {
 public:
  class Iterator {
   public:
    Iterator(
        std::deque<Scheduler::Payload>& q, std::deque<Scheduler::Payload>& dq)
        : at_dq_(false), citr_(q.begin()), end_itr_(q.end()),
          dqbegin_itr_(dq.begin()), dqend_itr_(dq.end())
    {
    }

    Iterator& operator++()
    {
      if ((++citr_ == end_itr_) && !at_dq_) {
        citr_ = dqbegin_itr_;
        end_itr_ = dqend_itr_;
        at_dq_ = true;
      }
      return *this;
    }

    Iterator& operator=(const Iterator& rhs)
    {
      at_dq_ = rhs.at_dq_;
      citr_ = rhs.citr_;
      end_itr_ = rhs.end_itr_;
      dqbegin_itr_ = rhs.dqbegin_itr_;
      dqend_itr_ = rhs.dqend_itr_;

      return *this;
    }

    bool operator!=(const Iterator& rhs)
    {
      return !(this->operator==(rhs));
    }

    bool operator==(const Iterator& rhs)
    {
      return (at_dq_ == rhs.at_dq_) && (citr_ == rhs.citr_);
    }

    Scheduler::Payload& operator*() { return *citr_; }

   private:
    bool at_dq_;
    std::deque<Scheduler::Payload>::iterator citr_;
    std::deque<Scheduler::Payload>::iterator end_itr_;
    std::deque<Scheduler::Payload>::iterator dqbegin_itr_;
    std::deque<Scheduler::Payload>::iterator dqend_itr_;
  };

  RequestQueue(const ModelQueuePolicy& policy)
      : action_(policy.action()),
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
    queue_.emplace_back(args);
    auto timeout_microseconds = default_timeout_microseconds_;
    if (allow_timeout_override_ && queue_.back()
                                           .request_provider_.RequestHeader()
                                           .timeout_microseconds() != 0) {
      timeout_microseconds = queue_.back()
                                 .request_provider_.RequestHeader()
                                 .timeout_microseconds();
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    timeout_queue_.emplace_back(
        TIMESPEC_TO_NANOS(now), timeout_microseconds * 1000);
    return Status::Success;
  }

  Scheduler::Payload Dequeue()
  {
    if (!queue_.empty()) {
      auto res = std::move(queue_.front());
      queue_.pop_front();
      timeout_queue_.pop_front();
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
    auto& dst_queue = action_ == DELAY ? delayed_queue_ : rejected_queue_;
    for (auto idx = 0; idx != queue_.size();) {
      if ((timeout_queue_[idx].second != 0) &&
          (now_nanoseconds - timeout_queue_[idx].first >
           timeout_queue_[idx].second)) {
        dst_queue.emplace_back(std::move(queue_[idx]));
        queue_.erase(queue_.begin() + idx);
        timeout_queue_.erase(timeout_queue_.begin() + idx);
      } else {
        // only advance the index if the current item is still valid
        idx++;
      }
    }
  }

  std::deque<Scheduler::Payload> ReleaseRejectedQueue()
  {
    std::deque<Scheduler::Payload> res;
    rejected_queue_.swap(res);
    return res;
  }

  bool Empty() { return Size() == 0; }

  size_t Size() { return queue_.size() + delayed_queue_.size(); }

  Iterator Begin()
  {
    return Iterator(queue_, delayed_queue_);
  }

  const Iterator& End()
  {
    return end_itr_;
  }

 private:
  std::deque<Scheduler::Payload> queue_;
  std::deque<uint64_t, uint64_t> timeout_queue_;
  std::deque<Scheduler::Payload> delayed_queue_;
  std::deque<Scheduler::Payload> rejected_queue_;
  Iterator end_itr_;
  const TimeoutAction action_;
  const uint64_t default_timeout_microseconds_;
  const bool allow_timeout_override_;
  const uint32_t max_queue_size_;
};

class PriorityQueue {
 public:
  using PriorityQueues = std::map<uint32_t, RequestQueue>;

  class Iterator {
   public:
    Iterator(PriorityQueues::iterator qitr, RequestQueue::Iterator ritr) : qitr_(qitr), ritr_(ritr) {}

    Iterator& operator++()
    {
      if ((++ritr_) == qitr_->second.End()) {
        ++qitr_;
        ritr_ = qitr_->second.Begin();
      }
      return *this;
    }

    bool operator!=(const Iterator& rhs)
    {
      return !(*this==(rhs));
    }

    bool operator==(const Iterator& rhs)
    {
      return (qitr_ == rhs.qitr_) && (ritr_ == rhs.ritr_);
    }

    Scheduler::Payload& operator*() { return *ritr_; }

    Scheduler::Payload* operator->() { return &(*ritr_); }

   private:
    PriorityQueues::iterator qitr_;
    RequestQueue::Iterator ritr_;
  };

  template <typename... Args>
  Status Emplace(uint32_t priority_level, Args&&... args)
  {
    return queues_[priority_level].Emplace(args);
  }

  Scheduler::Payload Dequeue()
  {
    for (auto& queue : queues_) {
      if (!queue.second.Empty()) {
        return queue.second.Dequeue();
      }
    }
  }

  size_t Size()
  {
    size_t res = 0;
    for (auto& queue : queues_) {
      res += queue.second.Size();
    }
  }

  bool Empty() { return Size() == 0; }

  Iterator&
  PendingIterator() {
    return pending_iterator_;
  }

  Iterator Begin()
  {
    return Iterator(
        queues_.begin(), queues_.begin()->second.Begin());
  }

  Iterator End()
  {
    auto last_queue = queues_.end()--;
    return Iterator(last_queue, last_queue->second.End());
  }

 private:
  PriorityQueues queues_;
  Iterator pending_iterator_;
  bool is_iterator_valid_;
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
