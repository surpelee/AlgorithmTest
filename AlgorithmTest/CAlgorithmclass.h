#pragma once
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <Algorithm>
#include <stack>
#include <queue>
#include <deque>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <tuple>
#include <list>
#include <bitset>
#include <functional>
#include <math.h>

using namespace std;

struct good_dj {
	int v;
	int w;
	good_dj(int _v, int _w) :v(_v), w(_w) {}
};

//并查集
struct UnionFind {
	vector<int> ancestor;
	UnionFind(int n) {
		ancestor.resize(n);
		for (int i = 0; i < n; ++i)
			ancestor[i] = i;
	}

	int find(int index) {
		if (index != ancestor[index])
			ancestor[index] = find(ancestor[index]);
		return ancestor[index];
		//return index == ancestor[index] ? index : ancestor[index] = find(ancestor[index]);
	}

	void merge(int u, int v) {
		ancestor[find(u)] = find(v);
	}
};

struct MinTreePrim //最小生成树所用的结构
{
	int val;
	int lowCost;
	MinTreePrim(){}
	MinTreePrim(int _val,int _cost):val(_val),lowCost(_cost){}
};

struct Arc //最小生成树所用的结构
{
	int u;
	int v;
	int cost;//边的权值
	Arc() :u(0), v(0), cost(-1) {}
	Arc(int _u,int _v,int _c):u(_u),v(_v),cost(_c){}
};

struct good
{
	int w, v, s;
};

struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}
};

struct status {
	int val;
	ListNode *ptr;
	bool operator < (const status& rhs) const {
		return val > rhs.val;//重载小顶堆  // priority_queue默认是最大堆 堆重载符号为 <
	}
};

struct minpair {
	int i, j;
	minpair() {}
	minpair(int _i, int _j) :i(_i), j(_j) {}
	bool operator < (const minpair& rhs) const {
		return (i + j) < (rhs.i + rhs.j);
	}
};

struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode() : val(0),left(NULL),right(NULL){}
	TreeNode(int x) :val(x), left(NULL), right(NULL) {}
	TreeNode(int x, TreeNode* l, TreeNode* r) : val(x),left(l),right(r){}
};

struct DoubleListNode {
	int key, val;
	DoubleListNode* next;
	DoubleListNode* prev;
	DoubleListNode(int _key, int _val) :key(_key), val(_val), next(nullptr), prev(nullptr) {}
};

class TreeNode1
{
public:
	int val;
	TreeNode1* left;
	TreeNode1* right;
	TreeNode1* next;
	TreeNode1() :val(0), left(NULL), right(NULL), next(NULL) {}
	TreeNode1(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}
	TreeNode1(int _val, TreeNode1* _left, TreeNode1* _right, TreeNode1* _next) : val(_val), left(_left), right(_right), next(_next) {}
};

class Node {
public:
	int val;
	vector<Node*> neighbors;

	Node() {
		val = 0;
		neighbors = vector<Node*>();
	}

	Node(int _val) {
		val = _val;
		neighbors = vector<Node*>();
	}

	Node(int _val, vector<Node*> _neighbors) {
		val = _val;
		neighbors = _neighbors;
	}
};

class Node_random {
public:
	int val;
	Node_random* next;
	Node_random* random;

	Node_random(int _val) {
		val = _val;
		next = NULL;
		random = NULL;
	}
};
/*LRUCache机制的实现 使用c++库list双向链表  LRUCache2使用自己构建的双向链表 实现原理:选择最久未使用的页面予以淘汰*/
class LRUCache {
public:
	LRUCache(int capacity) {
		this->cap = capacity;
	}

	int get(int key) {
		if (amap.find(key) == amap.end())//如果访问的key值不存在 返回-1
			return -1;
		pair<int, int> l = *amap[key];//key值存在 放到队头 并更新map中的位置
		cache.erase(amap[key]);
		cache.push_front(l);
		amap[key] = cache.begin();
		return l.second;
	}

	void put(int key, int value) {
		if (amap.find(key) == amap.end()) {//如果没找到key 数据已满就删除最后一个节点，将新的数据添加到头部
			if (cache.size() == cap) {	   //如果未满 删除链表中的节点，更新map
				pair<int, int> _end = cache.back();
				amap.erase(_end.first);
				cache.pop_back();
			}
			cache.push_front(make_pair(key, value));
			amap[key] = cache.begin();
		}
		else {
			cache.erase(amap[key]);
			cache.push_front(make_pair(key, value));
			amap[key] = cache.begin();
		}
	}
private:
	int cap;
	list<pair<int, int>> cache;
	unordered_map<int, list<pair<int, int>>::iterator> amap;
};

class DoubleList
{
public:
	DoubleList() {
		this->head = new DoubleListNode(0, 0);
		this->tail = new DoubleListNode(0, 0);
		this->head->next = this->tail;
		this->tail->prev = this->head;
		this->sSize = 0;
	}

	~DoubleList() {
		delete this->head;
		delete this->tail;
	}

	void erase(DoubleListNode* cur) {//删除当前节点
		cur->prev->next = cur->next;
		cur->next->prev = cur->prev;
		this->sSize--;
	}

	void push_front(DoubleListNode* cur) {
		cur->next = head->next;
		cur->prev = head;
		head->next->prev = cur;
		head->next = cur;
		this->sSize++;
	}

	DoubleListNode* back() {
		if (tail->prev == head) return nullptr;
		return tail->prev;
	}

	void pop_back() {
		if (tail->prev == head) return;
		erase(tail->prev);
	}

	int size() { return this->sSize; }
private:
	int sSize;
	DoubleListNode* head;
	DoubleListNode* tail;
};

class LRUCache2 {
public:
	LRUCache2(int capacity) {
		this->cap = capacity;
	}

	int get(int key) {
		if (amap.find(key) == amap.end()) return -1;
		int v = amap[key]->val;
		put(key, v);
		return v;
	}

	void put(int key, int value) {
		DoubleListNode* newnode = new DoubleListNode(key, value);
		if (amap.find(key) == amap.end()) {
			if (cap == cache.size()) {
				amap.erase(cache.back()->key);
				cache.pop_back();
			}
		}
		else
			cache.erase(amap[key]);
		cache.push_front(newnode);
		amap[key] = newnode;
	}

private:
	int cap;
	DoubleList cache;
	unordered_map<int, DoubleListNode*> amap;
};
/*LFU缓存机制 实现原理:选择最不经常使用的页面予以淘汰*/
class LFUCache {
public:
	struct Node {
		int key, val, frep;
		Node(int _key, int _val, int _frep) :key(_key), val(_val), frep(_frep) {}
	};
public:
	LFUCache(int capacity) {
		this->cap = capacity;
		this->min_frep = 0;
	}

	int get(int key) {
		if (cap == 0) return -1;
		if (key_amap.find(key) == key_amap.end())
			return -1;
		Node temp = *key_amap[key];
		int f = temp.frep;
		frep_amap[f].erase(key_amap[key]);
		if (frep_amap[f].size() == 0) {
			frep_amap.erase(f);
			if (min_frep == f)
				min_frep++;
		}
		frep_amap[f + 1].push_front(Node(temp.key, temp.val, f + 1));
		key_amap[key] = frep_amap[f + 1].begin();
		return temp.val;
	}

	void put(int key, int value) {
		if (cap == 0) return;
		if (key_amap.find(key) == key_amap.end()) {
			if (key_amap.size() == cap) {
				Node delete_end = frep_amap[min_frep].back();
				frep_amap[min_frep].pop_back();
				if (frep_amap[min_frep].size() == 0) {
					frep_amap.erase(min_frep);
				}
				key_amap.erase(delete_end.key);
			}
			frep_amap[1].push_front(Node(key, value, 1));
			key_amap[key] = frep_amap[1].begin();
			min_frep = 1;
		}
		else {
			Node temp = *key_amap[key];
			int f = temp.frep;
			frep_amap[f].erase(key_amap[key]);
			if (frep_amap[f].size() == 0) {
				frep_amap.erase(f);
				if (min_frep == f)
					min_frep++;
			}
			frep_amap[f + 1].push_front(Node(key, value, f + 1));
			key_amap[key] = frep_amap[f + 1].begin();
		}
	}
private:
	int cap;
	int min_frep;
	unordered_map<int, list<Node>::iterator> key_amap;
	unordered_map<int, list<Node>> frep_amap;
};

class MinStack {
public:
	/** initialize your data structure here. */
	MinStack() {

	}

	void push(int x) {
		s1.push(x);
		if (s2.empty() || s2.top() >= x)
			s2.push(x);
	}

	void pop() {
		if (s2.top() == s1.top()) s2.pop();
		s1.pop();
	}

	int top() {
		return s1.top();
	}

	int getMin() {
		return s2.top();
	}
private:
	stack<int> s1;
	stack<int> s2;
};

class BSTIterator {
public:
	BSTIterator(TreeNode* root) {
		while (root) {
			s.push(root);
			root = root->left;
		}
	}

	/** @return the next smallest number */
	int next() {
		TreeNode* temp = s.top(); s.pop();
		int res = temp->val;
		temp = temp->right;
		while (temp)
		{
			s.push(temp);
			temp = temp->left;
		}
		return res;
	}

	/** @return whether we have a next smallest number */
	bool hasNext() {
		return s.size() > 0 ? true : false;
	}
private:
	stack<TreeNode*> s;
};
//设计一个简易版的推特
class Twitter {
public:
	/** Initialize your data structure here. */
	Twitter() {
		recentMax = 10;
	}

	/** Compose a new tweet. */
	void postTweet(int userId, int tweetId) {
		cache.push_back(make_pair(userId, tweetId));
	}

	/** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
	vector<int> getNewsFeed(int userId) {
		vector<int> ans;
		for (int i = cache.size() - 1; recentMax > 0 && i >= 0; --i)
		{
			int temp = cache[i].first;
			if (temp == userId || amap[userId].find(temp) != amap[userId].end()) {
				ans.push_back(cache[i].second);
				recentMax--;
			}
		}
		return ans;
	}

	/** Follower follows a followee. If the operation is invalid, it should be a no-op. */
	void follow(int followerId, int followeeId) {
		amap[followerId].insert(followeeId);
	}

	/** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
	void unfollow(int followerId, int followeeId) {
		amap[followerId].erase(followeeId);
	}
private:
	int recentMax;
	vector<pair<int, int>> cache;
	unordered_map<int, unordered_set<int>> amap;
};
//使用链表+哈希
class Twitter2 {
	struct Node
	{
		unordered_set<int> followee;//关注的人的ID
		list<int> alist;//	用链表存储推特
	};
public:
	/** Initialize your data structure here. */
	Twitter2() {
		recentMax = 10;
		time = 0;
		user.clear();
	}

	void init(int tweetId) {
		user[tweetId].followee.clear();
		user[tweetId].alist.clear();
	}

	/** Compose a new tweet. */
	void postTweet(int userId, int tweetId) {
		if (user.find(userId) == user.end()) init(userId);
		if (user[userId].alist.size() == recentMax)
			user[userId].alist.pop_back();
		user[userId].alist.push_front(tweetId);
		tweetTime[tweetId] = ++time;
	}

	/** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
	vector<int> getNewsFeed(int userId) {
		vector<int> ans;
		for (list<int>::iterator i = user[userId].alist.begin(); i != user[userId].alist.end(); ++i)
			ans.push_back(*i);
		for (int temp_id : user[userId].followee) {
			if (temp_id == userId) continue;
			vector<int> res;
			list<int>::iterator it = user[temp_id].alist.begin();
			int i = 0;
			while (i < ans.size() && it != user[temp_id].alist.end()) {
				if (tweetTime[ans[i]] > tweetTime[*it]) {
					res.push_back(ans[i]);
					++i;
				}
				else {
					res.push_back(*it);
					++it;
				}
				if (res.size() == recentMax) break;
			}
			for (; i < (int)ans.size() && (int)res.size() < recentMax; ++i) res.push_back(ans[i]);
			for (; it != user[temp_id].alist.end() && (int)res.size() < recentMax; ++it) res.push_back(*it);
			ans.assign(res.begin(), res.end());
		}
		return ans;
	}

	/** Follower follows a followee. If the operation is invalid, it should be a no-op. */
	void follow(int followerId, int followeeId) {
		if (user.find(followerId) == user.end()) init(followerId);
		if (user.find(followeeId) == user.end()) init(followeeId);
		user[followerId].followee.insert(followeeId);
	}

	/** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
	void unfollow(int followerId, int followeeId) {
		user[followerId].followee.erase(followeeId);
	}
private:
	int recentMax, time;
	unordered_map<int, Node> user;
	unordered_map<int, int> tweetTime;//推特对应发送的时间
};
//实现Trie 前缀树  // 前缀树可以用在 1、自动补全  2、拼音检查 3、IP路由（最长前缀匹配） 4、九宫格 打字预测 5、单词匹配
class Trie {
public:
	/** Initialize your data structure here. */
	Trie() {
		str = "";
		isEnd = false;
		memset(next, 0, sizeof(next));
	}
	/** Inserts a word into the trie. */
	void insert(string word) {
		Trie* node = this;
		for (auto& c : word) {
			int temp = c - 'a';
			if (node->next[temp] == nullptr)
				node->next[temp] = new Trie();
			node = node->next[temp];
		}
		node->isEnd = true;
	}

	/** Returns if the word is in the trie. */
	bool search(string word) {
		Trie* node = this;
		for (auto& c : word) {
			node = node->next[c - 'a'];
			if (node == nullptr)
				return false;
		}
		return node->isEnd;
	}

	/** Returns if there is any word in the trie that starts with the given prefix. */
	bool startsWith(string prefix) {
		Trie* node = this;
		for (auto& c : prefix) {
			node = node->next[c - 'a'];
			if (node == nullptr)
				return false;
		}
		return true;
	}

public:
	bool isEnd;
	string str;
	Trie* next[26];
};

class Trie2 {
public:
	Trie2() {
		memset(next, 0, sizeof(next));
	}

	int insert(string word) {
		Trie2* node = this;
		bool isNew = false;
		for (int i = word.size() - 1; i >= 0; --i)
		{
			int temp = word[i] - 'a';
			if (node->next[temp] == nullptr) {
				isNew = true;
				node->next[temp] = new Trie2();
			}
			node = node->next[temp];
		}
		return isNew ? word.size() + 1 : 0;
	}

private:
	Trie2* next[26];
};

class WordDictionary {
public:
	/** Initialize your data structure here. */
	WordDictionary() {
		isEnd = false;
		memset(next, 0, sizeof(next));
	}

	/** Adds a word into the data structure. */
	void addWord(string word) {
		WordDictionary* node = this;
		for (auto& c : word)
		{
			int temp = c - 'a';
			if (node->next[temp] == nullptr)
				node->next[temp] = new WordDictionary();
			node = node->next[temp];
		}
		node->isEnd = true;
	}

	/** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
	bool search(string word) {
		WordDictionary* node = this;
		return seach_helper(word, node);
	}

	bool seach_helper(string word, WordDictionary* root) {
		WordDictionary* node = root;
		for (int i = 0; i < word.size(); ++i) {
			if (word[i] == '.') {
				for (int j = 0; j < 26; ++j)
				{
					if (node->next[j] != nullptr) {
						if (seach_helper(word.substr(i + 1), node->next[j]))
							return 1;
					}
				}
				return 0;
			}
			node = node->next[word[i] - 'a'];
			if (node == nullptr)
				return 0;
		}
		return node->isEnd;
	}

private:
	bool isEnd;
	WordDictionary* next[26];
};
//用队列实现栈
class MyStack {
public:
	/** Initialize your data structure here. */
	MyStack() {

	}

	/** Push element x onto stack. */
	void push(int x) {
		if (q1.empty()) q1.push(x);
		else {
			q1.push(x);
			int sSize = q1.size();
			while (sSize > 1)
			{
				int temp = q1.front();
				q1.pop();
				q1.push(temp);
				sSize--;
			}
		}
	}

	/** Removes the element on top of the stack and returns that element. */
	int pop() {
		int ans = top();
		q1.pop();
		return ans;
	}

	/** Get the top element. */
	int top() {
		return q1.front();
	}

	/** Returns whether the stack is empty. */
	bool empty() {
		return q1.empty();
	}
private:
	queue<int> q1;
};
//用栈实现队列
class MyQueue {
public:
	/** Initialize your data structure here. */
	MyQueue() {

	}

	/** Push element x to the back of queue. */
	void push(int x) {
		s1.push(x);
	}

	/** Removes the element from in front of queue and returns that element. */
	int pop() {
		int t = peek();
		s2.pop();
		return t;
	}

	/** Get the front element. */
	int peek() {
		if (s2.empty()) {
			while (!s1.empty())
			{
				int temp = s1.top();
				s1.pop();
				s2.push(temp);
			}
		}
		return s2.top();
	}

	/** Returns whether the queue is empty. */
	bool empty() {
		return s1.empty() && s2.empty();
	}
private:
	stack<int> s1, s2;
};
//队列中的最大值
class MaxQueue {
public:
	MaxQueue() {
		this->alist.clear();
		this->amap.clear();
	}

	int max_value() {
		if (alist.empty())
			return -1;
		auto it = amap.end();
		pair<int, list<int>::iterator> temp = *--it;
		return temp.first;
	}

	void push_back(int value) {
		alist.push_back(value);
		amap[value] = alist.end();
	}

	int pop_front() {
		if (alist.empty())
			return -1;
		int f = alist.front();
		alist.pop_front();
		amap.erase(f);
		return f;
	}
private:
	list<int> alist;
	map<int, list<int>::iterator> amap;
};
//队列中的最大值2
class MaxQueue2 {
public:
	MaxQueue2() {

	}

	int max_value() {
		if (dq.empty()) return -1;
		return dq.front();
	}

	void push_back(int value) {
		while (!dq.empty() && dq.back() < value)
			dq.pop_back();
		dq.push_back(value);
		q.push(value);
	}

	int pop_front() {
		if (q.empty()) return -1;
		int ans = q.front();
		if (ans == dq.front())
			dq.pop_front();
		q.pop();
		return ans;
	}
private:
	queue<int> q;
	deque<int> dq;
};
//数据流的中位数 使用最大堆和最小堆
class MedianFinder {
public:
	/** initialize your data structure here. */
	MedianFinder() {

	}

	void addNum(int num) {
		l.push(num);
		r.push(l.top());
		l.pop();
		if (l.size() < r.size()) {
			l.push(r.top());
			r.pop();
		}
	}

	double findMedian() {
		return l.size() > r.size() ? (double)l.top() : (r.top() + l.top())*0.5;
	}
private:
	priority_queue<int> l;//最大堆
	priority_queue<int, vector<int>, greater<int> > r;//最小堆
};
//数据流的中位数 使用multiset 迭代器进行
class MedianFinder2 {
public:
	/** initialize your data structure here. */
	MedianFinder2() {
		this->l = data.end();
		this->r = data.end();
	}

	void addNum(int num) {
		int n = data.size();
		data.insert(num);
		if (!n)
			l = r = data.begin();
		else if (n % 2) {
			if (num < *l)
				l--;
			else
				r++;
		}
		else {
			if (num > *l&&num < *r) {
				l++;
				r--;
			}
			else if (num >= *r)
				l++;
			else
				l = --r;
		}
	}

	double findMedian() {
		return (*l + *r)*0.5;
	}
private:
	multiset<int> data;
	multiset<int>::iterator l, r;
};
//二叉树的序列化和反序列化
class Codec {
public:

	// Encodes a tree to a single string.
	string serialize(TreeNode* root) {
		ostringstream out;
		serialize(root, out);
		return out.str();
	}

	// Decodes your encoded data to tree.
	TreeNode* deserialize(string data) {
		istringstream in(data);
		return deserialize(in);
	}
private:
	void serialize(TreeNode* root, ostringstream& out) {
		if (root) {
			out << root->val << ' ';
			serialize(root->left, out);
			serialize(root->right, out);
		}
		else {
			out << "# ";
		}

	}
	TreeNode* deserialize(istringstream& in) {
		string val;
		in >> val;
		if (val == "#") {
			return nullptr;
		}
		TreeNode* root = new TreeNode(stoi(val));
		root->left = deserialize(in);
		root->right = deserialize(in);
		return root;
	}
};
//区域和检索 数组不可变
class NumArray {
public:
	NumArray(vector<int>& nums) {
		this->sum.resize(nums.size() + 1);
		for (int i = 0; i < nums.size(); ++i)
			this->sum[i + 1] = this->sum[i] + nums[i];
	}

	int sumRange(int i, int j) {
		return sum[j + 1] - sum[i];
	}
private:
	vector<int> sum;
};
//区域和检索 数组可变  线段树
class NumArray1 {
public:
	NumArray1(vector<int>& nums) : n(nums.size()), tree(n << 1) {
		for (int i = n, j = 0; i < n << 1; ++i, ++j)
			tree[i] = nums[j];
		for (int i = n - 1; i > 0; --i)
			tree[i] = tree[i << 1] + tree[(i << 1) + 1];
	}
	void update(int pos, int val) {
		pos += n;
		val -= tree[pos];
		while (pos) {
			tree[pos] += val;
			pos >>= 1;
		}
	}
	int sumRange(int left, int right) {
		int res = 0;
		for (left += n, right += n; left <= right; left >>= 1, right >>= 1) {
			if (left & 1) res += tree[left++];
			if (!(right & 1)) res += tree[right--];
		}
		return res;
	}
private:
	int n;
	vector<int> tree;
};
//二维区域和检索 数组不可变
class NumMatrix {
public:
	NumMatrix(vector<vector<int>>& matrix) {
		if (!matrix.size() || !matrix[0].size()) return;
		int row = matrix.size(), col = matrix[0].size();
		this->sum.resize(row + 1, vector<int>(col + 1, 0));
		for (int i = 0; i < row; ++i) {
			for (int j = 0; j < col; ++j)
				sum[i + 1][j + 1] = matrix[i][j] + sum[i + 1][j] + sum[i][j + 1] - sum[i][j];
		}
	}

	int sumRegion(int row1, int col1, int row2, int col2) {
		return sum[row2 + 1][col2 + 1] - sum[row2 + 1][col1] - sum[row1][col2 + 1] + sum[row1][col1];
	}
private:
	vector<vector<int>> sum;
};
//线段树节点
struct SegmentTreeNode {
	int start;
	int end;
	int count;
	SegmentTreeNode* left;
	SegmentTreeNode* right;
	SegmentTreeNode(int _start, int _end) :start(_start), end(_end) {
		count = 0;
		left = NULL;
		right = NULL;
	}
};
//线段树的操作
class SegmentTree {
public:
	SegmentTree() {

	}

	SegmentTreeNode* build(int start, int end) {
		if (start > end)
			return nullptr;
		SegmentTreeNode* root = new SegmentTreeNode(start, end);
		if (start == end)
			root->count = 0;
		else {
			int mid = start + (end - start) / 2;
			root->left = build(start, mid);
			root->right = build(mid + 1, end);
		}
		return root;
	}

	int count(SegmentTreeNode* root, int start, int end) {
		if (root == NULL || start > end)
			return 0;
		if (start == root->start && end == root->end)
			return root->count;
		int mid = root->start + (root->end - root->start) / 2;
		int leftcount = 0, rightcount = 0;
		if (start <= mid) {
			if (mid < end)
				leftcount = count(root->left, start, mid);
			else
				leftcount = count(root->left, start, end);
		}
		if (mid < end) {
			if (start <= mid)
				rightcount = count(root->right, mid + 1, end);
			else
				rightcount = count(root->right, start, end);
		}
		return (leftcount + rightcount);
	}

	void insert(SegmentTreeNode* root, int index, int val) {
		if (root->start == index && root->end == index) {
			root->count += val;
			return;
		}
		int mid = root->start + (root->end - root->start) / 2;
		if (index >= root->start && index <= mid)
			insert(root->left, index, val);
		if (index > mid && index <= root->end)
			insert(root->right, index, val);
		root->count = root->left->count + root->right->count;
	}
};
//常数时间插入、删除和获取随机元素
class RandomizedSet {
public:
	/** Initialize your data structure here. */
	RandomizedSet() {
		res.clear();
		IndexMap.clear();
	}

	/** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
	bool insert(int val) {
		if (IndexMap.find(val) != IndexMap.end()) return false;
		res.push_back(val);
		IndexMap[val] = res.size() - 1;
		return true;
	}

	/** Removes a value from the set. Returns true if the set contained the specified element. */
	bool remove(int val) {
		if (IndexMap.find(val) == IndexMap.end()) return false;
		int len = res.size();
		int tmp = IndexMap[val];
		IndexMap[res[len - 1]] = tmp;
		IndexMap.erase(val);
		swap(res[tmp], res[res.size() - 1]);
		res.pop_back();
		return true;
	}

	/** Get a random element from the set. */
	int getRandom() {
		int len = res.size();
		int num = rand() % len;
		return res[num];
	}

private:
	unordered_map<int, int> IndexMap;
	vector<int> res;
};

namespace clionGitHub {
	int threeSumClosest(vector<int>& nums, int target);
	int countSubstrings(string s);
	int minSubArrayLen(int s, vector<int>& nums);
	int findKthLargest(vector<int>& nums, int k);
	int findLength(vector<int>& A, vector<int>& B);
	int kthSmallest(vector<vector<int>>& matrix, int k);//有序矩阵中第K小的元素
	bool searchMatrix(vector<vector<int>>& matrix, int target);//搜索二维矩阵2
	TreeNode* sortedArrayToBST(vector<int>& nums);//将有序数组转换为二叉树
	bool patternMatching(string pattern, string value);//模式匹配
	int longestValidParentheses(string s);//最长有效括号
	int maxScoreSightseeingPair(vector<int>& A);//最佳观光组合
	int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid);//不同路径2
	bool hasPathSum(TreeNode* root, int sum);//路径总和
	vector<int> divingBoard(int shorter, int longer, int k);//跳水板
	int respace(vector<string>& dictionary, string sentence);//恢复空格
	int maxProfit_freeze(vector<int>& prices);//买卖股票最佳时机包含冷冻期
	int maxProfit(vector<int>& prices);//买卖股票最佳时机
	int maxProfit2(vector<int>& prices);//买卖股票最佳时机 多次买卖股票
	int maxProfit3(vector<int>& prices);//买卖股票最佳时机 最多两次买卖股票
	int maxProfit4(int k, vector<int>& prices);//买卖股票最佳时机 最多k次买卖股票
	vector<int> countSmaller(vector<int>& nums);//计算右侧小于当前元素的个数
	int calculateMinimumHP(vector<vector<int>>& dungeon);//地下城游戏
	int numIdenticalPairs(vector<int>& nums);//好数对的个数
	int numSub(string s);//仅含 1 的子串数
	double maxProbability(int n, vector<vector<int>>& edges, vector<double>& succProb, int start, int end);//概率最大的路径

	void back_respace(unordered_set<string> &dictionary, string &sentence, int wordLen, int x, int num);
	bool back_hasPathSum(TreeNode *node, int sum, int ans);
	void CountPalin(const string& s, int l, int r);
	bool check_searchMatrix(vector<vector<int>>& matrix, int mid, int i, int j);
	TreeNode* back_sortedArrayToBST(vector<int>& nums, int l, int r);

	void maxHeapify(vector<int>& a, int i, int heapSize);
	void buildMaxHeap(vector<int>& a, int heapSize);

	void mergeCountSmaller(vector<int> &nums, vector<int> &index, vector<int> &res, int l, int r);

	//int m_int_clion;
}

class CAlgorithmclass
{
public:
	CAlgorithmclass();
	~CAlgorithmclass();

	vector<vector<int>> threeSum(vector<int>& nums);
	int mySqrt(int x);//平方根
	int climbStairs(int n);//爬楼梯
	string simplifyPath(string path);//简化路径
	vector<string> fullJustify(vector<string>& words, int maxWidth);//左右文本对齐
	int minDistance(string word1, string word2);//编辑距离
	void setZeroes(vector<vector<int>>& matrix);//矩阵置零
	bool searchMatrix(vector<vector<int>>& matrix, int target);//搜索二维矩阵
	void sortColors(vector<int>& nums);//颜色分类
	string minWindow(string s, string t);//最小覆盖子串
	vector<vector<int>> combine(int n, int k);//组合
	vector<vector<int>> subsets(vector<int>& nums);//子集
	bool exist(vector<vector<char>>& board, string word);//单词搜索
	vector<string> findWords(vector<vector<char>>& board, vector<string>& words);//单词搜索
	int removeDuplicates(vector<int>& nums);//删除重复数组2
	bool search(vector<int>& nums, int target);//搜索旋转排序数组2
	ListNode* deleteDuplicates(ListNode* head);//删除排序链表中的重复元素2
	ListNode* deleteDuplicates1(ListNode* head);//删除排序链表中重复元素
	int largestRectangleArea(vector<int>& heights);//柱状图中最大的矩形
	int maximalRectangle(vector<vector<char>>& matrix);//最大矩形
	ListNode* partition(ListNode* head, int x);//分隔链表
	bool isScramble(string s1, string s2);//扰乱字符
	void merge(vector<int>& nums1, int m, vector<int>& nums2, int n);//合并两个有序数组
	void solveSudoku(vector<vector<char>>& board);//解数独
	vector<int> grayCode(int n);//格雷编码
	vector<vector<int>> subsetsWithDup(vector<int>& nums);//子集2
	int numDecodings(string s);//解码方式
	ListNode* reverseBetween(ListNode* head, int m, int n);//反转链表2
	vector<string> restoreIpAddresses(string s);//复原IP地址
	vector<int> preorderTraversal(TreeNode* root);//二叉树的前序遍历 //迭代的方法
	vector<int> inorderTraversal(TreeNode* root);//二叉树的中序遍历
	vector<int> postorderTraversal(TreeNode* root);//二叉树的后序遍历 // 迭代
	vector<TreeNode*> generateTrees(int n);//不同的二叉搜索树2
	int numTrees(int n);//不同的二叉搜索树
	bool isInterleave(string s1, string s2, string s3);//交错字符串
	bool isValidBST(TreeNode* root);//验证二叉搜索树
	bool isSameTree(TreeNode* p, TreeNode* q);//相同的树
	void recoverTree(TreeNode* root);//恢复二叉搜索树
	bool isSymmetric(TreeNode* root);//镜像二叉树
	vector<vector<int>> levelOrder(TreeNode* root);//二叉树的层次遍历
	vector<vector<int>> zigzagLevelOrder(TreeNode* root);//二叉树的锯式层次遍历
	int maxDepth(TreeNode* root);//二叉树深度
	TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder);//从前序和中序构造二叉树
	TreeNode* buildTree1(vector<int>& inorder, vector<int>& postorder);//从中序和后序构造二叉树
	vector<vector<int>> levelOrderBottom(TreeNode* root);//二叉树层次遍历2
	TreeNode* sortedArrayToBST(vector<int>& nums);//将有序数组转换为二叉树
	TreeNode* sortedListToBST(ListNode* head);//将有序链表转换为二叉树
	bool isBalanced(TreeNode* root);//判断是否为平衡二叉树
	int minDepth(TreeNode* root);//二叉树最小深度
	bool hasPathSum(TreeNode* root, int sum);//路径总和
	vector<vector<int>> pathSum(TreeNode* root, int sum);//路径总和2
	void flatten(TreeNode* root);//二叉树展开为链表
	int numDistinct(string s, string t);//不同的子序列
	TreeNode1* connect(TreeNode1* root);//填充每个节点的下一个右侧节点指针
	TreeNode1* connect2(TreeNode1* root);//填充每个节点的下一个右侧节点指针2
	vector<vector<int>> generate(int numRows);//杨辉三角
	vector<int> getRow(int rowIndex);//杨辉三角2
	int minimumTotal(vector<vector<int>>& triangle);//三角形最小路径之和
	int maxProfit(vector<int>& prices);//买卖股票的最佳时机
	int maxProfit2(vector<int>& prices);//买卖股票的最佳时机2
	int maxProfit3(vector<int>& prices);//买卖股票的最佳时机3 //股票问题有模板思路
	int maxProfit4(int k, vector<int>& prices);//买卖股票的最佳时机4 k次交易
	int maxProfit5(vector<int>& prices);//买卖股票的最佳时机4 //含冷冻期
	int maxPathSum(TreeNode* root);//二叉树的最大路径之和   //利用后续遍历 自底向上获取左右节点的最大值
	bool isPalindrome(string s);//验证回文串
	void maxword(string& str);
	int ladderLength(string beginWord, string endWord, vector<string>& wordList);//单词接龙
	vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList);//单词接龙2
	int longestConsecutive(vector<int>& nums);//最长连续序列
	int sumNumbers(TreeNode* root);//求根到叶子结点的数字之和
	void yasuosuanfa(string& s);//腾讯试题1
	vector<int> kandaodelou(vector<int>& x);//试题2
	int xiuxi(vector<int>& gs, vector<bool>& jsf, int n);//试题3
	void linshi(int k, vector<vector<int>>& s);//复试题目 给定一个矩阵，是否能找到target
	void solve(vector<vector<char>>& board);//被围绕的区域
	vector<vector<string>> partition(string s);//分割回文串
	int minCut(string s);//分割回文串2
	Node* cloneGraph(Node* node);//克隆图    //深拷贝
	int canCompleteCircuit(vector<int>& gas, vector<int>& cost);//加油站
	int candy(vector<int>& ratings);//分发糖果
	int singleNumber(vector<int>& nums);//只出现一次的数字
	int singleNumber2(vector<int>& nums);//只出现一次的数字2
	vector<int> singleNumber3(vector<int>& nums);//只出现一次的数字3
	Node_random* copyRandomList(Node_random* head);//复制带随机指针的链表
	bool wordBreak(string s, vector<string>& wordDict);//单词拆分
	vector<string> wordBreak2(string s, vector<string>& wordDict);//单词拆分2  // 动态规划+回溯
	bool hasCycle(ListNode *head);//环形链表
	ListNode *detectCycle(ListNode *head);//环形链表2
	void reorderList(ListNode* head);//重排链表
	ListNode* insertionSortList(ListNode* head);//对链表进行插入排序
	ListNode* sortList(ListNode* head);//排序链表  //归并排序链表  可使用递归或者迭代
	int maxPoints(vector<vector<int>>& points);//直线上最多的点
	int evalRPN(vector<string>& tokens);//逆波兰表达式求值   //atoi标准函数将字符转换为数字
	string reverseWords(string s);//反转字符串中的单词
	int maxProduct(vector<int>& nums);//乘积最大子数组
	int findMin(vector<int>& nums);//寻找旋转排序数组中的最小值
	int findMin2(vector<int>& nums);//寻找旋转排序数组中的最小值   //数组中包含重复值
	ListNode *getIntersectionNode(ListNode *headA, ListNode *headB);//相交链表
	int findPeakElement(vector<int>& nums);//寻找峰值
	int compareVersion(string version1, string version2);//版本号比较
	string fractionToDecimal(int numerator, int denominator);//分数到小数
	vector<int> twoSum(vector<int>& numbers, int target);//两数之和2 输入是有序数组
	string convertToTitle(int n);//excel表列名称
	int majorityElement(vector<int>& nums);//多数元素
	int titleToNumber(string s);//excel表列名称
	int trailingZeroes(int n);//阶乘后有多少0
	int calculateMinimumHP(vector<vector<int>>& dungeon);//地下城游戏
	string largestNumber(vector<int>& nums);//最大数  //string的对象a和b比较  是逐个按子项的assic码比较
	vector<string> findRepeatedDnaSequences(string s);//重复DNA序列
	void rotate(vector<int>& nums, int k);//旋转数组
	uint32_t reverseBits(uint32_t n);//位运算 颠倒二进制位
	int hammingWeight(uint32_t n);//位1的个数  汉明重量
	vector<double> intersection(vector<int>& start1, vector<int>& end1, vector<int>& start2, vector<int>& end2);//交点
	int rob(vector<int>& nums);//打家劫舍
	vector<int> rightSideView(TreeNode* root);//二叉树的右视图
	int numIslands(vector<vector<char>>& grid);//岛屿的数量
	int rangeBitwiseAnd(int m, int n);//数字的范围按位与  //位运算应该有的思路
	bool isHappy(int n);//快乐数
	ListNode* removeElements(ListNode* head, int val);//移除链表元素
	int countPrimes(int n);//计算质数
	ListNode* addTwoNumbers2(ListNode* l1, ListNode* l2);//两数相加2
	bool isIsomorphic(string s, string t);//同构字符串
	bool canFinish(int numCourses, vector<vector<int>>& prerequisites);//课程表
	vector<vector<int>> updateMatrix(vector<vector<int>>& matrix);//矩阵
	int minSubArrayLen(int s, vector<int>& nums);//长度最小的子数组
	vector<vector<int>> merge(vector<vector<int>>& intervals);//合并区间
	vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites);//课程表2
	bool canJump(vector<int>& nums);//跳跃游戏
	int rob2(vector<int>& nums);//打家劫舍2
	int rob3(TreeNode* root);//打家劫舍3
	int maxArea(vector<int>& height);//盛水最多的容器
	string shortestPalindrome(string s);//最短回文串
	int findKthLargest(vector<int>& nums, int k);//数组中第K个最大元素
	int getMaxRepetitions(string s1, int n1, string s2, int n2);//统计重复个数
	vector<vector<int>> combinationSum3(int k, int n);//组合总和3
	bool containsDuplicate(vector<int>& nums);//存在重复元素
	bool containsNearbyDuplicate(vector<int>& nums, int k);//存在重复元素2
	bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t);//存在重复元素2
	vector<vector<int>> getSkyline(vector<vector<int>>& buildings);//天际线问题   //未完成
	int maximalSquare(vector<vector<char>>& matrix);//最大正方形
	int countNodes(TreeNode* root);//完全二叉树的节点个数
	int computeArea(int A, int B, int C, int D, int E, int F, int G, int H);//矩形面积
	int numberOfSubarrays(vector<int>& nums, int k);//统计优美子数组
	int calculate(string s);//基本计算器
	TreeNode* invertTree(TreeNode* root);//反转二叉树
	int calculate2(string s);//基本计算器2
	vector<string> summaryRanges(vector<int>& nums);//汇总区间
	vector<int> majorityElement2(vector<int>& nums);//求众数2 //摩尔投票法
	int kthSmallest(TreeNode* root, int k);//二叉树搜索树中第K小元素
	bool isPowerOfTwo(int n);//2的幂次方
	int countDigitOne(int n);//数字1的个数
	bool isPalindrome2(ListNode* head);//回文链表
	TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);//二叉搜索树的最近公共祖先
	TreeNode* lowestCommonAncestor2(TreeNode* root, TreeNode* p, TreeNode* q);//二叉树的最近公共祖先
	void deleteNode(ListNode* node);//删除链表中的节点
	vector<int> productExceptSelf(vector<int>& nums);//除自身以外数组的乘积
	vector<int> maxSlidingWindow(vector<int>& nums, int k);//滑动窗口最大值
	int waysToChange(int n);//硬币
	bool searchMatrix2(vector<vector<int>>& matrix, int target);//搜索二维矩阵2
	int reversePairs(vector<int>& nums);//数组中的逆序对 //归并排序
	vector<int> diffWaysToCompute(string input);//为运算表达式设计优先级
	bool isAnagram(string s, string t);//有效的字母异位词
	vector<string> binaryTreePaths(TreeNode* root);//二叉树的所有路径
	int addDigits(int num);//各位相加
	vector<vector<int>> permute(vector<int>& nums);//全排列
	bool isUgly(int num);//丑数
	int nthUglyNumber(int n);//丑数2
	int missingNumber(vector<int>& nums);//缺失的数字
	int movingCount(int m, int n, int k);//机器人的运动范围
	void rotate(vector<vector<int>>& matrix);//旋转矩阵
	void gameOfLife(vector<vector<int>>& board);//生命游戏
	ListNode* mergeKLists(vector<ListNode*>& lists);//合并K个排序链表  //分治 // 堆排序 最小堆  最大堆
	vector<int> maxDepthAfterSplit(string seq);//有效括号的嵌套深度
	int search2(vector<int>& nums, int target);//搜索旋转排序数组
	int superEggDrop(int K, int N);//鸡蛋掉落 智力题
	int hIndex(vector<int>& citations);//h指数
	int hIndex2(vector<int>& citations);//h指数2
	int numSquares(int n);//完全平方数
	vector<string> addOperators(string num, int target);//给表达式添加运算符
	vector<int> singleNumbers(vector<int>& nums);//数组中数字出现的次数
	void moveZeroes(vector<int>& nums);//移动零
	int findDuplicate(vector<int>& nums);//寻找重复的数
	bool wordPattern(string pattern, string str);//单词规律
	bool canWinNim(int n);//Nim游戏
	int lengthOfLIS(vector<int>& nums);//最长上升数组
	int coinChange(vector<int>& coins, int amount);//零钱兑换
	bool canMeasureWater(int x, int y, int z);//水壶问题    辗转相除法 求最大公因数  函数 gcd()
	int findInMountainArray(int target, vector<int>& nums);//山脉数组中查找目标值
	int longestPalindrome(string s);//最长回文串
	int diameterOfBinaryTree(TreeNode* root);//二叉树的直径
	int maxAreaOfIsland(vector<vector<int>>& grid);//岛屿的最大面积
	int minimumLengthEncoding(vector<string>& words);//单词的压缩编码
	bool isRectangleOverlap(vector<int>& rec1, vector<int>& rec2);//矩形重叠
	int surfaceArea(vector<vector<int>>& grid);//三维形体的表面积
	bool hasGroupsSizeX(vector<int>& deck);//卡牌分组
	int minIncrementForUnique(vector<int>& A);//使数组唯一的最小增量
	int orangesRotting(vector<vector<int>>& grid);//腐烂的橘子
	int numRookCaptures(vector<vector<char>>& board);//国际象棋可以一步被捕获的棋子
	bool canThreePartsEqualSum(vector<int>& A);//将数组分成和相等的三等分
	string gcdOfStrings(string str1, string str2);//字符串中的最大公因子
	vector<int> distributeCandies(int candies, int num_people);//分糖果2
	int countCharacters(vector<string>& words, string chars);//拼写单词merge
	int maxDistance(vector<vector<int>>& grid);//地图分析
	string compressString(string S);//字符串压缩
	int massage(vector<int>& nums);//按摩师
	vector<vector<int>> findContinuousSequence(int target);//和为s的连续正整数序列
	int lastRemaining(int n, int m);//圆圈中最后剩下的数字
	int jump(vector<int>& nums);//跳跃游戏
	string getHint(string secret, string guess);//哈希表
	vector<string> removeInvalidParentheses(string s);//删除无效括号
	bool isValidParentheses(string s);//有效的括号
	int mincostTickets(vector<int>& days, vector<int>& costs);//最低票价
	bool isAdditiveNumber(string num);//累加数
	bool isSubtree(TreeNode* s, TreeNode* t);//另一个树的子树
	vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges);//最小高度数   入度表 拓扑排序  无向图
	int maxCoins(vector<int>& nums);//戳气球
	int nthSuperUglyNumber(int n, vector<int>& primes);//超级丑数
	vector<int> countSmaller(vector<int>& nums);//计算右侧小于当前元素的个数
	string removeDuplicateLetters(string s);//去除重复字母
	int maxProduct1(vector<string>& words);//最大单词长度乘积
	int bulbSwitch(int n);//灯泡开关
	void wiggleSort(vector<int>& nums);//摆动排序
	double myPow(double x, int n);//
	int countRangeSum(vector<int>& nums, int lower, int upper);//区间和的个数
	ListNode* oddEvenList(ListNode* head);//奇偶链表
	int longestIncreasingPath(vector<vector<int>>& matrix);//最长递增序列
	int subarraySum(vector<int>& nums, int k);//数组中和为k的连续子数组的个数
	ListNode* reverseKGroup(ListNode* head, int k);//k个一组的反转列表
	bool isValidSerialization(string preorder);//验证二叉树的前序序列化
	vector<string> findItinerary(vector<vector<string>>& tickets);//重新安排行程
	bool increasingTriplet(vector<int>& nums);//递增的三元子序列
	bool validPalindrome(string s);//验证回文串2
	int findTheLongestSubstring(string s);//每个元音包含偶数次的最长子字符串    前缀和 状态压缩  位运算 哈希表
	vector<int> countBits(int num);//比特位计数
	double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2);//寻找两个正序数组的中位数
	vector<int> topKFrequent(vector<int>& nums, int k);//前K个高频元素
	string decodeString(string s);//字符串解码
	int subarraysDivByK(vector<int>& A, int K);//和为K的子数组的个数
	bool canPartition(vector<int>& nums);//分割等和子集
	int pathSum3(TreeNode* root, int sum);//路径总和3
	vector<int> findAnagrams(string s, string p);//找到字符串中所有字母异位词
	vector<int> findDisappearedNumbers(vector<int>& nums);//找到所有数组中消失的数字
	int hammingDistance(int x, int y);//汉明距离
	int findTargetSumWays(vector<int>& nums, int S);//目标和
	int findUnsortedSubarray(vector<int>& nums);//最短无序连续子数组
	TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2);//合并二叉树
	int leastInterval(vector<char>& tasks, int n);//任务调度器
	int translateNum(int num);
	vector<int> dailyTemperatures(vector<int>& T);//每日温度
	ListNode* removeDuplicateNodes(ListNode* head);//移除重复点
	bool robot(string command, vector<vector<int>>& obstacles, int x, int y);//机器人大冒险
	bool isMatch(string s, string p);//通配符匹配
	bool isMatch2(string s, string p);//正则表达式匹配
	int findBestValue(vector<int>& arr, int target);//
	int maxCoins2(vector<int>& nums);//戳气球
	int splitArray(vector<int>& nums, int m);//分割数组的最大值
	vector<int> smallestRange(vector<vector<int>>& nums);//最小区间
	string addStrings(string num1, string num2);//字符串相加
	vector<vector<int>> palindromePairs(vector<string>& words);//回文对
	int longestPalindromeSubseq(string s);//最长回文子序列
	int longestSubstring(string s, int k);//至少有K个重复字符的最长子串   //字符串递归分治
	int countBinarySubstrings(string s);//重复出现的子串 计算他们出现的次数
	int getSum(int a, int b);//两整数之和
	vector<vector<int>> fourSum(vector<int>& nums, int target);//四数之和
	int longestValidParentheses(string s);//最长有效括号
	string multiply(string num1, string num2);//字符串相乘
	int removeBoxes(vector<int>& boxes);//移除盒子
	vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor);//图像渲染
	//vector<int> bonus(int n, vector<vector<int>>& leadership, vector<vector<int>>& operations);//发LeeCoin
	int minCount(vector<int>& coins);//拿硬币
	int countSubstrings(string s);//回文字符串
	int numWays(int n, vector<vector<int>>& relation, int k);//传递信息
	vector<int> getTriggerTime(vector<vector<int>>& increase, vector<vector<int>>& requirements);//触发剧情
	vector<vector<char>> updateBoard(vector<vector<char>>& board, vector<int>& click);//扫雷游戏
	bool repeatedSubstringPattern(string s);//重复的子字符串
	vector<vector<int>> findSubsequences(vector<int>& nums);//递增子序列
	vector<string> letterCombinations(string digits);//电话号码的字母组合
	int findCircleNum(vector<vector<int>>& M);//朋友圈
	int minJump(vector<int>& jump);//最小跳跃次数
	bool canVisitAllRooms(vector<vector<int>>& rooms);//钥匙和房间
	bool PredictTheWinner(vector<int>& nums);//预测玩家
	bool isNumber(string s);
	vector<vector<int>> combinationSum(vector<int>& candidates, int target);//组合总和
	int minTime(vector<int>& time, int m);//小张刷题计划
	vector<int> maxNumber(vector<int>& nums1, vector<int>& nums2, int k);//拼接最大数
	bool isSelfCrossing(vector<int>& x);//路径交叉
	vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k);//和最小的k对数字
	vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges);//冗余连接 II
	int minCameraCover(TreeNode* root);//监控二叉树  //////////////////////////////////////////////////////比较有意思的二叉树，可多次看
	TreeNode* insertIntoBST(TreeNode* root, int val);//二叉树的插入
	int maxDistance(vector<int>& position, int m);//两球之间的磁力
	int minDays(int n);//吃掉N个橘子的最少天数
	int minimumOperations(string leaves);//秋叶收藏集
	vector<int> sumOfDistancesInTree(int N, vector<vector<int>>& edges);//树中距离之和
	int getMinimumDifference(TreeNode* root);//二叉搜索树的最小绝对差
	vector<int> partitionLabels(string S);//划分字母区间
	void nextPermutation(vector<int>& nums);//下一个排列
	int findRotateSteps(string ring, string key);//自由之路    动态规划
	string removeKdigits(string num, int k);//移掉K位数字
	int findMinArrowShots(vector<vector<int>>& points);//用最少数量的箭引爆气球
	int eraseOverlapIntervals(vector<vector<int>>& intervals);//无重叠区域

private:
	void dfs_getMinimumDifference(TreeNode* root);
	bool maxDistance_check(int mid, vector<int>& p, int m);
	int help_minCameraCover(TreeNode* root);
	bool check_minTime(vector<int>& time, int m, int mid);
	void back_combinationSum(vector<int>& candidates, vector<int>& res,int target,int index);
	void back_findCircleNum1(vector<vector<int>>& M, vector<bool>& visit, int a);
	int back_findCircleNum(vector<int>& res, int i);
	void back_letterCombinations(vector<string>& ans,vector<string>& hash,string& digits,string tmp,int a);
	void back_findSubsequences(vector<int>& nums, vector<int>& tmp, int cur, int pro);
	int back_countSubstrings(string & s, int l, int r);
	int back_removeBoxs(vector<int>& boxes, int l, int r,int k,vector<vector<vector<int>>>& dp);
	bool isPalindrome(string& a,string& b);
	int solve_maxCoins2(int l, int r);
	void back_findTargetSumWays(vector<int>& nums, int a, int S, int sum);
	void back_pathSum3(TreeNode* root, int sum, int level, vector<int> &temp);
	bool back_canPartition(int num, unordered_map<int, int>&vis1, unordered_map<int, int>&vis2, vector<int>& nums);
	string back_decode(string s, int a);
	vector<int> back_rob3(TreeNode* root);
	bool back_validPalindrome(string& s, int l, int r, bool once);
	void back_longestIncreasingPath(vector<vector<int>>& matrix, vector<vector<int>>& sign, int i, int j);
	int back_countRangeSum(vector<long long> &S, vector<long long> &assist, int L, int R, int low, int up);
	double help_mypow(double x, int n);
	void back_countSmaller(vector<int>& nums, int l, int r, vector<int>& temp, vector<int>& index);
	int back_maxCoins(vector<int>& nums, int l, int r, vector<vector<int>>& dp);
	bool back_isSubtree(TreeNode * s, TreeNode * t);
	bool back_isAdditiveNumber(string num, vector<long double> tmp);
	bool orEmpty(unordered_map<char, int> &amap);
	void backCombine(int n, int k, vector<int> temp, int a);
	void backSubsets(vector<int>& nums, vector<int>& temp, int a);
	void backExist(vector<vector<char>>& board, vector<vector<int>>& _or, string word, int i, int j, int n, int m, int k);
	bool dfs(vector<vector<char>>& board, string& word, int size, int x, int y, vector<vector<int>>& f);
	void back_dfs(vector<vector<char>>& board, int i, int j, Trie* node);
	void DFS_shudu(int i, int j, vector<vector<char>>& board);//解数独
	void backsubsetsWithDup(vector<int>& nums, vector<int>& temp, int a);
	void back_IpAddresses(string s, int n, string segment);
	void back_IpAddresses2(string & s, int k, int a, string res);
	void middle_order(TreeNode* root);//中序遍历
	vector<TreeNode*> back_generateTrees(int start, int end);
	int back_numTrees(int start, int end);
	bool back_inInterleave(string& s1, string& s2, string& s3, int i, int j, int k);
	bool back_isSymmetric(TreeNode* left, TreeNode* right);
	void back_maxDepth(TreeNode* root, int level);
	TreeNode* back_buildTree(vector<int>& preorder, int l, int r);
	TreeNode* back_buildTree1(int l, int r);
	TreeNode* back_sortedArrayToBST(vector<int> nums, int l, int r);
	TreeNode* back_sortedListToBST(ListNode* head, ListNode* end);
	bool back_isBalanced(TreeNode* root, int& height);//判断节点是否为平衡节点
	int back_minDepth(TreeNode* root);
	bool back_hasPathSum(TreeNode* root, int cur);
	void back_PathSum(TreeNode* root, int cur, vector<int>& curVt);
	void back_flatten(TreeNode* root);
	void back_numDistinct(string& s, string& t, int i, int j);
	void back_connect(TreeNode1* root);
	void back_connect2(TreeNode1* root);
	int back_maxPathSum(TreeNode* root);
	bool isconvert(string& s1, string& s2);//判断两个字符串是否可以转换
	bool back_findLadders(unordered_set<string>& words, unordered_set<string>& beginset, unordered_set<string>& endset, bool isend);
	void back_print_findLadders(string &beginWord, string &endWord, vector<string>& strVt);
	void back_sumNumbers(TreeNode* root, int cur);
	void back_solve(int i, int j, int row, int col, vector<vector<char>>& board);
	bool is_partition(string& s, int a, int i);//是否为回文字符串
	void back_partition(string& s, int a, vector<string>& temp, vector<vector<bool>>& dp);
	bool back_wordBreak(string& s, int a, vector<bool>& dp);
	void back_wordBreak2(string s, int a, vector<bool>& dp, string s1);
	ListNode* reverse_list(ListNode* head);//反转链表
	void SplitString(const string& s, const string& c, vector<string>& v);//分割字符串  c是分割的字符
	bool compair_string(string& s1, string& s2);//自定义字符串的比较
	uint32_t reverseByte(uint32_t _byte, unordered_map<uint32_t, uint32_t>& _map);//反转 字节
	int hamming_Weight(uint32_t _byte, unordered_map<uint32_t, int>& _map);
	bool intersection_inside(int x1, int y1, int x2, int y2, int xk, int yk);
	void intersection_update(vector<double>& ans, double xk, double yk);
	void back_rightSideView(TreeNode* root, int level);//递归深度搜索
	void back_numIslands(vector<vector<char>>& grid, int i, int j);
	int getnext(int n);
	void back_updateMatrix(vector<vector<int>>& matrix, int i, int j);//  邻接表和拓扑排序
	bool back_canJump(int a, vector<int>& nums);
	void back_combinationSum3(vector<vector<int>>& ans, vector<int>& res, int k, int n, int a);
	int findnumsTree(TreeNode* root);//查找树的子节点数
	bool back_lowestCommonAncestor(TreeNode* root, TreeNode * p, TreeNode * q);
	int mergeSort(vector<int>& nums, vector<int>& temp, int l, int r);
	void back_binaryTreePaths(TreeNode* root, string s);
	void back_permute(vector<int>& nums, vector<int>& temp, vector<bool>& visit);
	void back_movingCount(vector<vector<bool>>& visit, int i, int j, int k);
	int help_move(int i, int j);
	ListNode* mergeLists(vector<ListNode*>& lists, int l, int r);
	int back_calculateF(int k, int t);
	void back_addOperators(string num, int a, string resstr, long res, long multi, int target);
	void back_coinChange(vector<int>& coins, int a, int amount, int cur);
	int back_diameterOfBinaryTree(TreeNode* root);
	void back_maxAreaOfIsland(vector<vector<int>>& grid, int i, int j);

public:
	int gcd(int a, int b) {
		return !a ? b : gcd(b % a, a);
		/*
		//a ,b 不能为0
		while(b^=a^=b^=a%=b);
		return a;
		*/
	};//求最大公约数  辗转相除
	vector<int> sortArray(vector<int>& nums);//排序算法
	void QuickSort(vector<int>& nums, int low, int high);//快速排序算法   挖坑法
	void heapSort(vector<int>& nums);//堆排序 堆是最大堆 全排序
	vector<int> MaxKth_heapSort(vector<int>& nums, int k);//堆排序 堆是最大堆 最大的前k个值
	void buildMaxHeap(vector<int>& nums, int len);//构建最大堆
	void maxHeapify(vector<int>& nums, int i, int len);//最大堆的实现
	void MergeSort(vector<int>& nums, int l, int r);//归并排序

	vector<int> getLeastNumbers(vector<int>& arr, int k);//求前K个最小值
	void QuickSort_Select(vector<int>& arr, int start, int end, int k);//基于快速排序的变形

	bool KMPstring(const string& s, const string& t);//KMP字符串优化匹配

	//背包九讲
	int bag1(int N, int M, const vector<int>& w, const vector<int>& v);//01背包
	int bag2(int N, int M, const vector<int>& w, const vector<int>& v);//完全背包
	int bag3(int N, int M, const vector<int>& w, const vector<int>& v, const vector<int>& s);//多重背包1
	int bag4(int N, int M, const vector<int>& w, const vector<int>& v, const vector<int>& s);//多重背包2
	int bag5(int N, int M, const vector<int>& w, const vector<int>& v, const vector<int>& s);//混合背包
	int bag6(int N, int V, int M, const vector<int>& v, const vector<int>& m, const vector<int>& w);//二维费用的背包问题

	vector<pair<int, int>> MIniSpanTree_Kruskal(int cnt, vector<vector<int>>& adj);//最小生成树Kruskal
	bool FindTree(int u,int v,vector<vector<int>>& Tree);

	vector<pair<int, int>> MiniSpanTree_Prim(int cnt,vector<vector<int>>& adj, vector<MinTreePrim>& cntTree); //最小生成树Prim

	void Dijkstra(vector<vector<int>>& adj,vector<int>& dis,int v,int e);
	int networkDelayTime(vector<vector<int>>& times, int N, int K);//Dijkstra算法

	//二分查找
	//查找具体值
	int binary_search_lr(vector<int>& nums,int target) {
		int l = 0, r = nums.size() - 1;
		while (l <= r) {
			int mid = l + (r - l) / 2;
			if (target == nums[mid]) return mid;
			else if (nums[mid] > target) r = mid - 1;
			else if (nums[mid] < target) l = mid + 1;
		}
		return -1;
	}
	//查找左边界
	int binary_left(vector<int>& nums,int target) {
		int l = 0, r = nums.size() - 1;
		while (l <= r) {
			int mid = l + (r - l) / 2;
			if (nums[mid] == target) r = mid - 1;
			else if (nums[mid] > target) r = mid - 1;
			else if (nums[mid] < target) l = mid + 1;
		}
		if (l >= nums.size() && nums[l] != target) return -1;
		return l;
	}
	//查找右边界
	int binary_right(vector<int>& nums, int target) {
		int l = 0, r = nums.size() - 1;
		while (l <= r) {
			int mid = l + (r - l) / 2;
			if (nums[mid] == target) l = mid + 1;
			else if (nums[mid] > target) r = mid - 1;
			else if (nums[mid] < target) l = mid + 1;
		}
		if (r <= 0 && nums[r] != target) return -1;
		return r;
	}

private:
	int m_int, m_row, m_col;
	vector<int> m_intVt, m_index;
	vector<vector<int>> m_intVtVt;
	vector<vector<char>> m_charVtVt;
	vector<pair<int, int>> m_pairVt;
	bool m_bool;
	bool solved = false;
	bool row[10][10], col[10][10], box[10][10];//二维数组的行记录board一行或一列或一个宫
	vector<string> m_strVt;
	vector<TreeNode*> m_TreeVt;
	TreeNode* m_tree;
	unordered_map<int, int> m_map;
	vector<vector<string>> m_strVtVt;
	unordered_map<string, vector<string>> m_mmap;
	unordered_map<string, vector<int>> m_svmap;
	unordered_map<Node*, Node*> m_NodeMap;
	unordered_set<Node*> m_NodeSet;
	unordered_set<string> m_setstr;
	/*
	位运算常见的使用方法 与“&”，或“|”，异或“^”,非“~”
	1.n&(n-1)	表示将n的二进制最右边的1置为0
	2.n&(-n)	表示n和n的补码进行与运算,保留n的二进制最右边的1
	3."^"	异或运算,相同值异或后位0
	4.">>","<<"		右移和左移  分别表示n的二进制向右/左移几位,如果仅移1位相当于 n除2/乘2

	在位运算中，a^b表示无法进位的相加，a&b左移以为就表示进位
	*/
};

