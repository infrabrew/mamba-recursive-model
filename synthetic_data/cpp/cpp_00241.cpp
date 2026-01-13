#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

template<typename T>
class ThreadSafeQueue {{
private:
    mutable std::mutex mutex;
    std::queue<T> queue;
    std::condition_variable cond;

public:
    void push(T value) {{
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(std::move(value));
        cond.notify_one();
    }}

    bool try_pop(T& value) {{
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty()) {{
            return false;
        }}
        value = std::move(queue.front());
        queue.pop();
        return true;
    }}

    void wait_and_pop(T& value) {{
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this] {{ return !queue.empty(); }});
        value = std::move(queue.front());
        queue.pop();
    }}
}};