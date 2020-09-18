// AlgorithmTest.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include "CAlgorithmclass.h"
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

vector<int> finMax(const vector<int>& nums) {
	vector<int> ans(nums.size(), 0);
	ans[nums.size() - 1] = INT_MAX;
	stack<int> s;
	s.push(nums[nums.size() - 1]);
	for (int i = nums.size() - 2; i >= 0; --i) {
		while (!s.empty() && s.top() <= nums[i]) {
			s.pop();
		}
		ans[i] = s.top();
		s.push(nums[i]);
	}
	return ans;
}

class WordBook
{
public:
	WordBook() {
		Loction = -1;
		memset(ch, 0, sizeof(ch));
	}

	void insertNode(string& s, int Loc) {
		WordBook* node = this;
		for (int i = 0; i < s.size(); ++i) {
			int x = s[i] - 'a';
			if (node->ch[x] == nullptr) {
				node->ch[x] = new WordBook();
			}
			node = node->ch[x];
		}
		node->Loction = Loc;
	}

	int findNode(string& s, int i, int j) {
		WordBook* node = this;
		for (int k = j; k >= i; --k) {
			int x = s[k] - 'a';
			if (node->ch[x] == nullptr)
				return -1;
			node = node->ch[x];
		}
		return node->Loction;
	}

private:
	int Loction;
	WordBook* ch[26];
};

bool isPalindrom(string& s, int l, int r) {
	int len = r - l + 1;
	for (int i = 0; i < len / 2; ++i) {
		if (s[i + l] != s[r - i]) return false;
	}
	return true;
}

vector<vector<int>> palindromePairs(vector<string>& words) {
	vector<vector<int>> ans;
	WordBook* root = new WordBook();
	for (int i = 0; i < words.size(); ++i)
		root->insertNode(words[i], i);
	for (int i = 0; i < words.size(); ++i) {
		int len = words[i].size();
		for (int j = 0; j <= len; ++j) {
			if (isPalindrom(words[i], j, len - 1)) {
				int leftId = root->findNode(words[i], 0, j - 1);
				if (leftId != i && leftId != -1)
					ans.push_back({ i,leftId });
			}
			if (j && isPalindrom(words[i], 0, j - 1)) {
				int rightId = root->findNode(words[i], j, len - 1);
				if (rightId != i && rightId != -1)
					ans.push_back({ rightId,i });
			}
		}
	}
	return ans;
}

struct TreeInt {
	int l, r, len;
	TreeInt() :l(0), r(0), len(0) {}
	TreeInt(int _l, int _r, int _len) :l(_l), r(_r), len(_len) {}
	bool operator < (const TreeInt& osg) const {
		return this->len < osg.len;
	}
};

TreeInt center(string& s, int l, int r) {
	while (l >= 0 && r < s.size() && s[l] == s[r]) {
		--l;
		++r;
	}
	TreeInt one(l, r, r - l - 1);
	return one;
}

string longestPalindrome(string s) {
	TreeInt ans;
	for (int i = 0; i < s.size(); ++i) {
		TreeInt one = center(s, i, i);
		//ans = one.len > ans.len ? one : ans;
		ans = one < ans ? ans : one;
		TreeInt two = center(s, i, i + 1);
		//ans = two.len > ans.len ? two : ans;
		ans = two < ans ? ans : two;
	}
	return s.substr(ans.l + 1, ans.len);
}



int main() {

	vector<vector<int>> adjMat(6,vector<int>(6,-1));
	adjMat[0][1] = 6; adjMat[0][2] = 1; adjMat[0][3] = 5;
	adjMat[1][0] = 6; adjMat[1][2] = 5; adjMat[1][4] = 3;
	adjMat[2][0] = 1; adjMat[2][1] = 5; adjMat[2][3] = 5; adjMat[2][4] = 6; adjMat[2][5] = 4;
	adjMat[3][0] = 5; adjMat[3][2] = 5; adjMat[3][5] = 2;
	adjMat[4][1] = 3; adjMat[4][2] = 6; adjMat[4][5] = 6;
	adjMat[5][2] = 4; adjMat[5][3] = 2; adjMat[5][4] = 6;

	vector<MinTreePrim> cntTree;

	vector<vector<int>> edges = { {1,2},{2,3},{3,4},{4,1},{1,5} };
	CAlgorithmclass solve;
	auto ans = solve.findRedundantDirectedConnection(edges);
	
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
