// AlgorithmTest.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include "CAlgorithmclass.h"
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

bool helper(vector<int>& a, vector<int>& b) {
	unordered_map<int, int> amap;
	for (int i = 0; i < 6; ++i)
		amap[a[i]] = i;
	int change = 0;
	int rev = 0;
	for (int i = 0; i < 6; i += 2) {
		int t = amap[b[i]];
		if (t % 2 == 0) {
			if (a[t + 1] != b[i + 1])
				return false;
			if (t != i)
				change++;
		}
		else {
			if (a[t - 1] != b[i + 1])
				return false;
			change++;
			rev++;
		}
	}
	return (change / 2 + rev) % 2 ? false : true;
}

vector<int> solve(int N, vector<vector<int>>& nums) {
	vector<int> ans;
	vector<bool> visit(N, false);
	for (int i = 0; i < N; ++i) {
		if (visit[i]) continue;
		visit[i] = true;
		int res = 1;
		for (int j = i + 1; j < N; ++j) {
			if (visit[j]) continue;
			if (helper(nums[i], nums[j])) {
				visit[j] = true;
				res++;
			}
		}
		ans.push_back(res);
	}
	return ans;
}

int main() {
	int N = 2;
	CAlgorithmclass solve1;
	vector<vector<int>> nums = {};
	auto res = solve1.smallestRange(nums);
	return 0;
}





























// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
