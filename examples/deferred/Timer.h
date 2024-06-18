#pragma once
#include <chrono>

namespace resolutions
{
	typedef std::chrono::microseconds	microseconds;
	typedef std::chrono::milliseconds	milliseconds;
	typedef std::chrono::seconds			seconds;
}


template<typename Accuracy>
class Timer
{
public:
		
	Timer() : mStart(std::chrono::high_resolution_clock::now()), mLast(mStart)
	{}
	void restart()
	{
		mStart = std::chrono::high_resolution_clock::now();
	}
	long long total_elapsed()
	{
		return std::chrono::duration_cast<Accuracy>(std::chrono::high_resolution_clock::now() - mStart).count();
	}
	long long delta_elapsed()
	{
		const long long ret = std::chrono::duration_cast<Accuracy>(std::chrono::high_resolution_clock::now() - mLast).count();
		mLast = std::chrono::high_resolution_clock::now();
		return ret;
	}

private:
	std::chrono::high_resolution_clock::time_point mStart;
	std::chrono::high_resolution_clock::time_point mLast;
};