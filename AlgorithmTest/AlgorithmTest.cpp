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

void helper(vector<vector<char>>& board, vector<vector<int>>& dir, int i, int j) {
	board[i][j] = '#';
	for (int k = 0; k < 4; ++k) {
		int ti = dir[k][0] + i;
		int tj = dir[k][1] + j;
		if (ti < 0 || tj < 0 || ti >= board.size() || tj >= board[0].size() || board[ti][tj] != 'O') continue;
		helper(board, dir, ti, tj);
	}
}

void solve(vector<vector<char>>& board) {
	if (board.empty()) return;
	int m = board.size(), n = board[0].size();
	vector<vector<int>> dir = { {-1,0},{0,-1},{1,0},{0,1} };
	for (int i = 0; i < m; ++i) {
		if (board[i][0] == 'O') helper(board, dir, i, 0);
		if (board[i][n - 1] == 'O') helper(board, dir, i, n - 1);
	}
	for (int i = 0; i < n; ++i) {
		if (board[0][i] == 'O') helper(board, dir, 0, i);
		if (board[n - 1][i] == 'O') helper(board, dir, n - 1, i);
	}
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (board[i][j] == 'O') board[i][j] = 'X';
			if (board[i][j] == '#') board[i][j] = 'O';
		}
	}
	return;
}

int main() {

	vector<vector<char>> words = { {'X', 'O', 'X', 'O', 'X', 'O'},{'O', 'X', 'O', 'X', 'O', 'X'},{'X', 'O','X', 'O', 'X', 'O'},{'O', 'X', 'O', 'X', 'O', 'X'} };
	solve(words);
	return 0;

	int N = 2;
	CAlgorithmclass solve1;
	vector<int> nums = { 3,2,1,5,4,3,1,2,6 };
	string a = "00110";
	TreeNode* root = new TreeNode(1);
	root->left = new TreeNode(3);
	root->left->right = new TreeNode(2);

	auto res = solve1.countBinarySubstrings(a);
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
