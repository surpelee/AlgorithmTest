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

//���鼯
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

struct MinTreePrim //��С���������õĽṹ
{
	int val;
	int lowCost;
	MinTreePrim(){}
	MinTreePrim(int _val,int _cost):val(_val),lowCost(_cost){}
};

struct Arc //��С���������õĽṹ
{
	int u;
	int v;
	int cost;//�ߵ�Ȩֵ
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
		return val > rhs.val;//����С����  // priority_queueĬ�������� �����ط���Ϊ <
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
/*LRUCache���Ƶ�ʵ�� ʹ��c++��list˫������  LRUCache2ʹ���Լ�������˫������ ʵ��ԭ��:ѡ�����δʹ�õ�ҳ��������̭*/
class LRUCache {
public:
	LRUCache(int capacity) {
		this->cap = capacity;
	}

	int get(int key) {
		if (amap.find(key) == amap.end())//������ʵ�keyֵ������ ����-1
			return -1;
		pair<int, int> l = *amap[key];//keyֵ���� �ŵ���ͷ ������map�е�λ��
		cache.erase(amap[key]);
		cache.push_front(l);
		amap[key] = cache.begin();
		return l.second;
	}

	void put(int key, int value) {
		if (amap.find(key) == amap.end()) {//���û�ҵ�key ����������ɾ�����һ���ڵ㣬���µ�������ӵ�ͷ��
			if (cache.size() == cap) {	   //���δ�� ɾ�������еĽڵ㣬����map
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

	void erase(DoubleListNode* cur) {//ɾ����ǰ�ڵ�
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
/*LFU������� ʵ��ԭ��:ѡ�������ʹ�õ�ҳ��������̭*/
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
//���һ�����װ������
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
//ʹ������+��ϣ
class Twitter2 {
	struct Node
	{
		unordered_set<int> followee;//��ע���˵�ID
		list<int> alist;//	������洢����
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
	unordered_map<int, int> tweetTime;//���ض�Ӧ���͵�ʱ��
};
//ʵ��Trie ǰ׺��  // ǰ׺���������� 1���Զ���ȫ  2��ƴ����� 3��IP·�ɣ��ǰ׺ƥ�䣩 4���Ź��� ����Ԥ�� 5������ƥ��
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
//�ö���ʵ��ջ
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
//��ջʵ�ֶ���
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
//�����е����ֵ
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
//�����е����ֵ2
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
//����������λ�� ʹ�����Ѻ���С��
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
	priority_queue<int> l;//����
	priority_queue<int, vector<int>, greater<int> > r;//��С��
};
//����������λ�� ʹ��multiset ����������
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
//�����������л��ͷ����л�
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
//����ͼ��� ���鲻�ɱ�
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
//����ͼ��� ����ɱ�  �߶���
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
//��ά����ͼ��� ���鲻�ɱ�
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
//�߶����ڵ�
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
//�߶����Ĳ���
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
//����ʱ����롢ɾ���ͻ�ȡ���Ԫ��
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
	int kthSmallest(vector<vector<int>>& matrix, int k);//��������е�KС��Ԫ��
	bool searchMatrix(vector<vector<int>>& matrix, int target);//������ά����2
	TreeNode* sortedArrayToBST(vector<int>& nums);//����������ת��Ϊ������
	bool patternMatching(string pattern, string value);//ģʽƥ��
	int longestValidParentheses(string s);//���Ч����
	int maxScoreSightseeingPair(vector<int>& A);//��ѹ۹����
	int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid);//��ͬ·��2
	bool hasPathSum(TreeNode* root, int sum);//·���ܺ�
	vector<int> divingBoard(int shorter, int longer, int k);//��ˮ��
	int respace(vector<string>& dictionary, string sentence);//�ָ��ո�
	int maxProfit_freeze(vector<int>& prices);//������Ʊ���ʱ�������䶳��
	int maxProfit(vector<int>& prices);//������Ʊ���ʱ��
	int maxProfit2(vector<int>& prices);//������Ʊ���ʱ�� ���������Ʊ
	int maxProfit3(vector<int>& prices);//������Ʊ���ʱ�� �������������Ʊ
	int maxProfit4(int k, vector<int>& prices);//������Ʊ���ʱ�� ���k��������Ʊ
	vector<int> countSmaller(vector<int>& nums);//�����Ҳ�С�ڵ�ǰԪ�صĸ���
	int calculateMinimumHP(vector<vector<int>>& dungeon);//���³���Ϸ
	int numIdenticalPairs(vector<int>& nums);//�����Եĸ���
	int numSub(string s);//���� 1 ���Ӵ���
	double maxProbability(int n, vector<vector<int>>& edges, vector<double>& succProb, int start, int end);//��������·��

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
	int mySqrt(int x);//ƽ����
	int climbStairs(int n);//��¥��
	string simplifyPath(string path);//��·��
	vector<string> fullJustify(vector<string>& words, int maxWidth);//�����ı�����
	int minDistance(string word1, string word2);//�༭����
	void setZeroes(vector<vector<int>>& matrix);//��������
	bool searchMatrix(vector<vector<int>>& matrix, int target);//������ά����
	void sortColors(vector<int>& nums);//��ɫ����
	string minWindow(string s, string t);//��С�����Ӵ�
	vector<vector<int>> combine(int n, int k);//���
	vector<vector<int>> subsets(vector<int>& nums);//�Ӽ�
	bool exist(vector<vector<char>>& board, string word);//��������
	vector<string> findWords(vector<vector<char>>& board, vector<string>& words);//��������
	int removeDuplicates(vector<int>& nums);//ɾ���ظ�����2
	bool search(vector<int>& nums, int target);//������ת��������2
	ListNode* deleteDuplicates(ListNode* head);//ɾ�����������е��ظ�Ԫ��2
	ListNode* deleteDuplicates1(ListNode* head);//ɾ�������������ظ�Ԫ��
	int largestRectangleArea(vector<int>& heights);//��״ͼ�����ľ���
	int maximalRectangle(vector<vector<char>>& matrix);//������
	ListNode* partition(ListNode* head, int x);//�ָ�����
	bool isScramble(string s1, string s2);//�����ַ�
	void merge(vector<int>& nums1, int m, vector<int>& nums2, int n);//�ϲ�������������
	void solveSudoku(vector<vector<char>>& board);//������
	vector<int> grayCode(int n);//���ױ���
	vector<vector<int>> subsetsWithDup(vector<int>& nums);//�Ӽ�2
	int numDecodings(string s);//���뷽ʽ
	ListNode* reverseBetween(ListNode* head, int m, int n);//��ת����2
	vector<string> restoreIpAddresses(string s);//��ԭIP��ַ
	vector<int> preorderTraversal(TreeNode* root);//��������ǰ����� //�����ķ���
	vector<int> inorderTraversal(TreeNode* root);//���������������
	vector<int> postorderTraversal(TreeNode* root);//�������ĺ������ // ����
	vector<TreeNode*> generateTrees(int n);//��ͬ�Ķ���������2
	int numTrees(int n);//��ͬ�Ķ���������
	bool isInterleave(string s1, string s2, string s3);//�����ַ���
	bool isValidBST(TreeNode* root);//��֤����������
	bool isSameTree(TreeNode* p, TreeNode* q);//��ͬ����
	void recoverTree(TreeNode* root);//�ָ�����������
	bool isSymmetric(TreeNode* root);//���������
	vector<vector<int>> levelOrder(TreeNode* root);//�������Ĳ�α���
	vector<vector<int>> zigzagLevelOrder(TreeNode* root);//�������ľ�ʽ��α���
	int maxDepth(TreeNode* root);//���������
	TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder);//��ǰ��������������
	TreeNode* buildTree1(vector<int>& inorder, vector<int>& postorder);//������ͺ����������
	vector<vector<int>> levelOrderBottom(TreeNode* root);//��������α���2
	TreeNode* sortedArrayToBST(vector<int>& nums);//����������ת��Ϊ������
	TreeNode* sortedListToBST(ListNode* head);//����������ת��Ϊ������
	bool isBalanced(TreeNode* root);//�ж��Ƿ�Ϊƽ�������
	int minDepth(TreeNode* root);//��������С���
	bool hasPathSum(TreeNode* root, int sum);//·���ܺ�
	vector<vector<int>> pathSum(TreeNode* root, int sum);//·���ܺ�2
	void flatten(TreeNode* root);//������չ��Ϊ����
	int numDistinct(string s, string t);//��ͬ��������
	TreeNode1* connect(TreeNode1* root);//���ÿ���ڵ����һ���Ҳ�ڵ�ָ��
	TreeNode1* connect2(TreeNode1* root);//���ÿ���ڵ����һ���Ҳ�ڵ�ָ��2
	vector<vector<int>> generate(int numRows);//�������
	vector<int> getRow(int rowIndex);//�������2
	int minimumTotal(vector<vector<int>>& triangle);//��������С·��֮��
	int maxProfit(vector<int>& prices);//������Ʊ�����ʱ��
	int maxProfit2(vector<int>& prices);//������Ʊ�����ʱ��2
	int maxProfit3(vector<int>& prices);//������Ʊ�����ʱ��3 //��Ʊ������ģ��˼·
	int maxProfit4(int k, vector<int>& prices);//������Ʊ�����ʱ��4 k�ν���
	int maxProfit5(vector<int>& prices);//������Ʊ�����ʱ��4 //���䶳��
	int maxPathSum(TreeNode* root);//�����������·��֮��   //���ú������� �Ե����ϻ�ȡ���ҽڵ�����ֵ
	bool isPalindrome(string s);//��֤���Ĵ�
	void maxword(string& str);
	int ladderLength(string beginWord, string endWord, vector<string>& wordList);//���ʽ���
	vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList);//���ʽ���2
	int longestConsecutive(vector<int>& nums);//���������
	int sumNumbers(TreeNode* root);//�����Ҷ�ӽ�������֮��
	void yasuosuanfa(string& s);//��Ѷ����1
	vector<int> kandaodelou(vector<int>& x);//����2
	int xiuxi(vector<int>& gs, vector<bool>& jsf, int n);//����3
	void linshi(int k, vector<vector<int>>& s);//������Ŀ ����һ�������Ƿ����ҵ�target
	void solve(vector<vector<char>>& board);//��Χ�Ƶ�����
	vector<vector<string>> partition(string s);//�ָ���Ĵ�
	int minCut(string s);//�ָ���Ĵ�2
	Node* cloneGraph(Node* node);//��¡ͼ    //���
	int canCompleteCircuit(vector<int>& gas, vector<int>& cost);//����վ
	int candy(vector<int>& ratings);//�ַ��ǹ�
	int singleNumber(vector<int>& nums);//ֻ����һ�ε�����
	int singleNumber2(vector<int>& nums);//ֻ����һ�ε�����2
	vector<int> singleNumber3(vector<int>& nums);//ֻ����һ�ε�����3
	Node_random* copyRandomList(Node_random* head);//���ƴ����ָ�������
	bool wordBreak(string s, vector<string>& wordDict);//���ʲ��
	vector<string> wordBreak2(string s, vector<string>& wordDict);//���ʲ��2  // ��̬�滮+����
	bool hasCycle(ListNode *head);//��������
	ListNode *detectCycle(ListNode *head);//��������2
	void reorderList(ListNode* head);//��������
	ListNode* insertionSortList(ListNode* head);//��������в�������
	ListNode* sortList(ListNode* head);//��������  //�鲢��������  ��ʹ�õݹ���ߵ���
	int maxPoints(vector<vector<int>>& points);//ֱ�������ĵ�
	int evalRPN(vector<string>& tokens);//�沨�����ʽ��ֵ   //atoi��׼�������ַ�ת��Ϊ����
	string reverseWords(string s);//��ת�ַ����еĵ���
	int maxProduct(vector<int>& nums);//�˻����������
	int findMin(vector<int>& nums);//Ѱ����ת���������е���Сֵ
	int findMin2(vector<int>& nums);//Ѱ����ת���������е���Сֵ   //�����а����ظ�ֵ
	ListNode *getIntersectionNode(ListNode *headA, ListNode *headB);//�ཻ����
	int findPeakElement(vector<int>& nums);//Ѱ�ҷ�ֵ
	int compareVersion(string version1, string version2);//�汾�űȽ�
	string fractionToDecimal(int numerator, int denominator);//������С��
	vector<int> twoSum(vector<int>& numbers, int target);//����֮��2 ��������������
	string convertToTitle(int n);//excel��������
	int majorityElement(vector<int>& nums);//����Ԫ��
	int titleToNumber(string s);//excel��������
	int trailingZeroes(int n);//�׳˺��ж���0
	int calculateMinimumHP(vector<vector<int>>& dungeon);//���³���Ϸ
	string largestNumber(vector<int>& nums);//�����  //string�Ķ���a��b�Ƚ�  ������������assic��Ƚ�
	vector<string> findRepeatedDnaSequences(string s);//�ظ�DNA����
	void rotate(vector<int>& nums, int k);//��ת����
	uint32_t reverseBits(uint32_t n);//λ���� �ߵ�������λ
	int hammingWeight(uint32_t n);//λ1�ĸ���  ��������
	vector<double> intersection(vector<int>& start1, vector<int>& end1, vector<int>& start2, vector<int>& end2);//����
	int rob(vector<int>& nums);//��ҽ���
	vector<int> rightSideView(TreeNode* root);//������������ͼ
	int numIslands(vector<vector<char>>& grid);//���������
	int rangeBitwiseAnd(int m, int n);//���ֵķ�Χ��λ��  //λ����Ӧ���е�˼·
	bool isHappy(int n);//������
	ListNode* removeElements(ListNode* head, int val);//�Ƴ�����Ԫ��
	int countPrimes(int n);//��������
	ListNode* addTwoNumbers2(ListNode* l1, ListNode* l2);//�������2
	bool isIsomorphic(string s, string t);//ͬ���ַ���
	bool canFinish(int numCourses, vector<vector<int>>& prerequisites);//�γ̱�
	vector<vector<int>> updateMatrix(vector<vector<int>>& matrix);//����
	int minSubArrayLen(int s, vector<int>& nums);//������С��������
	vector<vector<int>> merge(vector<vector<int>>& intervals);//�ϲ�����
	vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites);//�γ̱�2
	bool canJump(vector<int>& nums);//��Ծ��Ϸ
	int rob2(vector<int>& nums);//��ҽ���2
	int rob3(TreeNode* root);//��ҽ���3
	int maxArea(vector<int>& height);//ʢˮ��������
	string shortestPalindrome(string s);//��̻��Ĵ�
	int findKthLargest(vector<int>& nums, int k);//�����е�K�����Ԫ��
	int getMaxRepetitions(string s1, int n1, string s2, int n2);//ͳ���ظ�����
	vector<vector<int>> combinationSum3(int k, int n);//����ܺ�3
	bool containsDuplicate(vector<int>& nums);//�����ظ�Ԫ��
	bool containsNearbyDuplicate(vector<int>& nums, int k);//�����ظ�Ԫ��2
	bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t);//�����ظ�Ԫ��2
	vector<vector<int>> getSkyline(vector<vector<int>>& buildings);//���������   //δ���
	int maximalSquare(vector<vector<char>>& matrix);//���������
	int countNodes(TreeNode* root);//��ȫ�������Ľڵ����
	int computeArea(int A, int B, int C, int D, int E, int F, int G, int H);//�������
	int numberOfSubarrays(vector<int>& nums, int k);//ͳ������������
	int calculate(string s);//����������
	TreeNode* invertTree(TreeNode* root);//��ת������
	int calculate2(string s);//����������2
	vector<string> summaryRanges(vector<int>& nums);//��������
	vector<int> majorityElement2(vector<int>& nums);//������2 //Ħ��ͶƱ��
	int kthSmallest(TreeNode* root, int k);//�������������е�KСԪ��
	bool isPowerOfTwo(int n);//2���ݴη�
	int countDigitOne(int n);//����1�ĸ���
	bool isPalindrome2(ListNode* head);//��������
	TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);//�����������������������
	TreeNode* lowestCommonAncestor2(TreeNode* root, TreeNode* p, TreeNode* q);//�������������������
	void deleteNode(ListNode* node);//ɾ�������еĽڵ�
	vector<int> productExceptSelf(vector<int>& nums);//��������������ĳ˻�
	vector<int> maxSlidingWindow(vector<int>& nums, int k);//�����������ֵ
	int waysToChange(int n);//Ӳ��
	bool searchMatrix2(vector<vector<int>>& matrix, int target);//������ά����2
	int reversePairs(vector<int>& nums);//�����е������ //�鲢����
	vector<int> diffWaysToCompute(string input);//Ϊ������ʽ������ȼ�
	bool isAnagram(string s, string t);//��Ч����ĸ��λ��
	vector<string> binaryTreePaths(TreeNode* root);//������������·��
	int addDigits(int num);//��λ���
	vector<vector<int>> permute(vector<int>& nums);//ȫ����
	bool isUgly(int num);//����
	int nthUglyNumber(int n);//����2
	int missingNumber(vector<int>& nums);//ȱʧ������
	int movingCount(int m, int n, int k);//�����˵��˶���Χ
	void rotate(vector<vector<int>>& matrix);//��ת����
	void gameOfLife(vector<vector<int>>& board);//������Ϸ
	ListNode* mergeKLists(vector<ListNode*>& lists);//�ϲ�K����������  //���� // ������ ��С��  ����
	vector<int> maxDepthAfterSplit(string seq);//��Ч���ŵ�Ƕ�����
	int search2(vector<int>& nums, int target);//������ת��������
	int superEggDrop(int K, int N);//�������� ������
	int hIndex(vector<int>& citations);//hָ��
	int hIndex2(vector<int>& citations);//hָ��2
	int numSquares(int n);//��ȫƽ����
	vector<string> addOperators(string num, int target);//�����ʽ��������
	vector<int> singleNumbers(vector<int>& nums);//���������ֳ��ֵĴ���
	void moveZeroes(vector<int>& nums);//�ƶ���
	int findDuplicate(vector<int>& nums);//Ѱ���ظ�����
	bool wordPattern(string pattern, string str);//���ʹ���
	bool canWinNim(int n);//Nim��Ϸ
	int lengthOfLIS(vector<int>& nums);//���������
	int coinChange(vector<int>& coins, int amount);//��Ǯ�һ�
	bool canMeasureWater(int x, int y, int z);//ˮ������    շת����� ���������  ���� gcd()
	int findInMountainArray(int target, vector<int>& nums);//ɽ�������в���Ŀ��ֵ
	int longestPalindrome(string s);//����Ĵ�
	int diameterOfBinaryTree(TreeNode* root);//��������ֱ��
	int maxAreaOfIsland(vector<vector<int>>& grid);//�����������
	int minimumLengthEncoding(vector<string>& words);//���ʵ�ѹ������
	bool isRectangleOverlap(vector<int>& rec1, vector<int>& rec2);//�����ص�
	int surfaceArea(vector<vector<int>>& grid);//��ά����ı����
	bool hasGroupsSizeX(vector<int>& deck);//���Ʒ���
	int minIncrementForUnique(vector<int>& A);//ʹ����Ψһ����С����
	int orangesRotting(vector<vector<int>>& grid);//���õ�����
	int numRookCaptures(vector<vector<char>>& board);//�����������һ�������������
	bool canThreePartsEqualSum(vector<int>& A);//������ֳɺ���ȵ����ȷ�
	string gcdOfStrings(string str1, string str2);//�ַ����е��������
	vector<int> distributeCandies(int candies, int num_people);//���ǹ�2
	int countCharacters(vector<string>& words, string chars);//ƴд����merge
	int maxDistance(vector<vector<int>>& grid);//��ͼ����
	string compressString(string S);//�ַ���ѹ��
	int massage(vector<int>& nums);//��Ħʦ
	vector<vector<int>> findContinuousSequence(int target);//��Ϊs����������������
	int lastRemaining(int n, int m);//ԲȦ�����ʣ�µ�����
	int jump(vector<int>& nums);//��Ծ��Ϸ
	string getHint(string secret, string guess);//��ϣ��
	vector<string> removeInvalidParentheses(string s);//ɾ����Ч����
	bool isValidParentheses(string s);//��Ч������
	int mincostTickets(vector<int>& days, vector<int>& costs);//���Ʊ��
	bool isAdditiveNumber(string num);//�ۼ���
	bool isSubtree(TreeNode* s, TreeNode* t);//��һ����������
	vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges);//��С�߶���   ��ȱ� ��������  ����ͼ
	int maxCoins(vector<int>& nums);//������
	int nthSuperUglyNumber(int n, vector<int>& primes);//��������
	vector<int> countSmaller(vector<int>& nums);//�����Ҳ�С�ڵ�ǰԪ�صĸ���
	string removeDuplicateLetters(string s);//ȥ���ظ���ĸ
	int maxProduct1(vector<string>& words);//��󵥴ʳ��ȳ˻�
	int bulbSwitch(int n);//���ݿ���
	void wiggleSort(vector<int>& nums);//�ڶ�����
	double myPow(double x, int n);//
	int countRangeSum(vector<int>& nums, int lower, int upper);//����͵ĸ���
	ListNode* oddEvenList(ListNode* head);//��ż����
	int longestIncreasingPath(vector<vector<int>>& matrix);//���������
	int subarraySum(vector<int>& nums, int k);//�����к�Ϊk������������ĸ���
	ListNode* reverseKGroup(ListNode* head, int k);//k��һ��ķ�ת�б�
	bool isValidSerialization(string preorder);//��֤��������ǰ�����л�
	vector<string> findItinerary(vector<vector<string>>& tickets);//���°����г�
	bool increasingTriplet(vector<int>& nums);//��������Ԫ������
	bool validPalindrome(string s);//��֤���Ĵ�2
	int findTheLongestSubstring(string s);//ÿ��Ԫ������ż���ε�����ַ���    ǰ׺�� ״̬ѹ��  λ���� ��ϣ��
	vector<int> countBits(int num);//����λ����
	double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2);//Ѱ�����������������λ��
	vector<int> topKFrequent(vector<int>& nums, int k);//ǰK����ƵԪ��
	string decodeString(string s);//�ַ�������
	int subarraysDivByK(vector<int>& A, int K);//��ΪK��������ĸ���
	bool canPartition(vector<int>& nums);//�ָ�Ⱥ��Ӽ�
	int pathSum3(TreeNode* root, int sum);//·���ܺ�3
	vector<int> findAnagrams(string s, string p);//�ҵ��ַ�����������ĸ��λ��
	vector<int> findDisappearedNumbers(vector<int>& nums);//�ҵ�������������ʧ������
	int hammingDistance(int x, int y);//��������
	int findTargetSumWays(vector<int>& nums, int S);//Ŀ���
	int findUnsortedSubarray(vector<int>& nums);//�����������������
	TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2);//�ϲ�������
	int leastInterval(vector<char>& tasks, int n);//���������
	int translateNum(int num);
	vector<int> dailyTemperatures(vector<int>& T);//ÿ���¶�
	ListNode* removeDuplicateNodes(ListNode* head);//�Ƴ��ظ���
	bool robot(string command, vector<vector<int>>& obstacles, int x, int y);//�����˴�ð��
	bool isMatch(string s, string p);//ͨ���ƥ��
	bool isMatch2(string s, string p);//������ʽƥ��
	int findBestValue(vector<int>& arr, int target);//
	int maxCoins2(vector<int>& nums);//������
	int splitArray(vector<int>& nums, int m);//�ָ���������ֵ
	vector<int> smallestRange(vector<vector<int>>& nums);//��С����
	string addStrings(string num1, string num2);//�ַ������
	vector<vector<int>> palindromePairs(vector<string>& words);//���Ķ�
	int longestPalindromeSubseq(string s);//�����������
	int longestSubstring(string s, int k);//������K���ظ��ַ�����Ӵ�   //�ַ����ݹ����
	int countBinarySubstrings(string s);//�ظ����ֵ��Ӵ� �������ǳ��ֵĴ���
	int getSum(int a, int b);//������֮��
	vector<vector<int>> fourSum(vector<int>& nums, int target);//����֮��
	int longestValidParentheses(string s);//���Ч����
	string multiply(string num1, string num2);//�ַ������
	int removeBoxes(vector<int>& boxes);//�Ƴ�����
	vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor);//ͼ����Ⱦ
	//vector<int> bonus(int n, vector<vector<int>>& leadership, vector<vector<int>>& operations);//��LeeCoin
	int minCount(vector<int>& coins);//��Ӳ��
	int countSubstrings(string s);//�����ַ���
	int numWays(int n, vector<vector<int>>& relation, int k);//������Ϣ
	vector<int> getTriggerTime(vector<vector<int>>& increase, vector<vector<int>>& requirements);//��������
	vector<vector<char>> updateBoard(vector<vector<char>>& board, vector<int>& click);//ɨ����Ϸ
	bool repeatedSubstringPattern(string s);//�ظ������ַ���
	vector<vector<int>> findSubsequences(vector<int>& nums);//����������
	vector<string> letterCombinations(string digits);//�绰�������ĸ���
	int findCircleNum(vector<vector<int>>& M);//����Ȧ
	int minJump(vector<int>& jump);//��С��Ծ����
	bool canVisitAllRooms(vector<vector<int>>& rooms);//Կ�׺ͷ���
	bool PredictTheWinner(vector<int>& nums);//Ԥ�����
	bool isNumber(string s);
	vector<vector<int>> combinationSum(vector<int>& candidates, int target);//����ܺ�
	int minTime(vector<int>& time, int m);//С��ˢ��ƻ�
	vector<int> maxNumber(vector<int>& nums1, vector<int>& nums2, int k);//ƴ�������
	bool isSelfCrossing(vector<int>& x);//·������
	vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k);//����С��k������
	vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges);//�������� II
	int minCameraCover(TreeNode* root);//��ض�����  //////////////////////////////////////////////////////�Ƚ�����˼�Ķ��������ɶ�ο�
	TreeNode* insertIntoBST(TreeNode* root, int val);//�������Ĳ���
	int maxDistance(vector<int>& position, int m);//����֮��Ĵ���
	int minDays(int n);//�Ե�N�����ӵ���������
	int minimumOperations(string leaves);//��Ҷ�ղؼ�
	vector<int> sumOfDistancesInTree(int N, vector<vector<int>>& edges);//���о���֮��
	int getMinimumDifference(TreeNode* root);//��������������С���Բ�
	vector<int> partitionLabels(string S);//������ĸ����
	void nextPermutation(vector<int>& nums);//��һ������
	int findRotateSteps(string ring, string key);//����֮·    ��̬�滮
	string removeKdigits(string num, int k);//�Ƶ�Kλ����
	int findMinArrowShots(vector<vector<int>>& points);//�����������ļ���������
	int eraseOverlapIntervals(vector<vector<int>>& intervals);//���ص�����

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
	void DFS_shudu(int i, int j, vector<vector<char>>& board);//������
	void backsubsetsWithDup(vector<int>& nums, vector<int>& temp, int a);
	void back_IpAddresses(string s, int n, string segment);
	void back_IpAddresses2(string & s, int k, int a, string res);
	void middle_order(TreeNode* root);//�������
	vector<TreeNode*> back_generateTrees(int start, int end);
	int back_numTrees(int start, int end);
	bool back_inInterleave(string& s1, string& s2, string& s3, int i, int j, int k);
	bool back_isSymmetric(TreeNode* left, TreeNode* right);
	void back_maxDepth(TreeNode* root, int level);
	TreeNode* back_buildTree(vector<int>& preorder, int l, int r);
	TreeNode* back_buildTree1(int l, int r);
	TreeNode* back_sortedArrayToBST(vector<int> nums, int l, int r);
	TreeNode* back_sortedListToBST(ListNode* head, ListNode* end);
	bool back_isBalanced(TreeNode* root, int& height);//�жϽڵ��Ƿ�Ϊƽ��ڵ�
	int back_minDepth(TreeNode* root);
	bool back_hasPathSum(TreeNode* root, int cur);
	void back_PathSum(TreeNode* root, int cur, vector<int>& curVt);
	void back_flatten(TreeNode* root);
	void back_numDistinct(string& s, string& t, int i, int j);
	void back_connect(TreeNode1* root);
	void back_connect2(TreeNode1* root);
	int back_maxPathSum(TreeNode* root);
	bool isconvert(string& s1, string& s2);//�ж������ַ����Ƿ����ת��
	bool back_findLadders(unordered_set<string>& words, unordered_set<string>& beginset, unordered_set<string>& endset, bool isend);
	void back_print_findLadders(string &beginWord, string &endWord, vector<string>& strVt);
	void back_sumNumbers(TreeNode* root, int cur);
	void back_solve(int i, int j, int row, int col, vector<vector<char>>& board);
	bool is_partition(string& s, int a, int i);//�Ƿ�Ϊ�����ַ���
	void back_partition(string& s, int a, vector<string>& temp, vector<vector<bool>>& dp);
	bool back_wordBreak(string& s, int a, vector<bool>& dp);
	void back_wordBreak2(string s, int a, vector<bool>& dp, string s1);
	ListNode* reverse_list(ListNode* head);//��ת����
	void SplitString(const string& s, const string& c, vector<string>& v);//�ָ��ַ���  c�Ƿָ���ַ�
	bool compair_string(string& s1, string& s2);//�Զ����ַ����ıȽ�
	uint32_t reverseByte(uint32_t _byte, unordered_map<uint32_t, uint32_t>& _map);//��ת �ֽ�
	int hamming_Weight(uint32_t _byte, unordered_map<uint32_t, int>& _map);
	bool intersection_inside(int x1, int y1, int x2, int y2, int xk, int yk);
	void intersection_update(vector<double>& ans, double xk, double yk);
	void back_rightSideView(TreeNode* root, int level);//�ݹ��������
	void back_numIslands(vector<vector<char>>& grid, int i, int j);
	int getnext(int n);
	void back_updateMatrix(vector<vector<int>>& matrix, int i, int j);//  �ڽӱ����������
	bool back_canJump(int a, vector<int>& nums);
	void back_combinationSum3(vector<vector<int>>& ans, vector<int>& res, int k, int n, int a);
	int findnumsTree(TreeNode* root);//���������ӽڵ���
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
		//a ,b ����Ϊ0
		while(b^=a^=b^=a%=b);
		return a;
		*/
	};//�����Լ��  շת���
	vector<int> sortArray(vector<int>& nums);//�����㷨
	void QuickSort(vector<int>& nums, int low, int high);//���������㷨   �ڿӷ�
	void heapSort(vector<int>& nums);//������ �������� ȫ����
	vector<int> MaxKth_heapSort(vector<int>& nums, int k);//������ �������� ����ǰk��ֵ
	void buildMaxHeap(vector<int>& nums, int len);//��������
	void maxHeapify(vector<int>& nums, int i, int len);//���ѵ�ʵ��
	void MergeSort(vector<int>& nums, int l, int r);//�鲢����

	vector<int> getLeastNumbers(vector<int>& arr, int k);//��ǰK����Сֵ
	void QuickSort_Select(vector<int>& arr, int start, int end, int k);//���ڿ�������ı���

	bool KMPstring(const string& s, const string& t);//KMP�ַ����Ż�ƥ��

	//�����Ž�
	int bag1(int N, int M, const vector<int>& w, const vector<int>& v);//01����
	int bag2(int N, int M, const vector<int>& w, const vector<int>& v);//��ȫ����
	int bag3(int N, int M, const vector<int>& w, const vector<int>& v, const vector<int>& s);//���ر���1
	int bag4(int N, int M, const vector<int>& w, const vector<int>& v, const vector<int>& s);//���ر���2
	int bag5(int N, int M, const vector<int>& w, const vector<int>& v, const vector<int>& s);//��ϱ���
	int bag6(int N, int V, int M, const vector<int>& v, const vector<int>& m, const vector<int>& w);//��ά���õı�������

	vector<pair<int, int>> MIniSpanTree_Kruskal(int cnt, vector<vector<int>>& adj);//��С������Kruskal
	bool FindTree(int u,int v,vector<vector<int>>& Tree);

	vector<pair<int, int>> MiniSpanTree_Prim(int cnt,vector<vector<int>>& adj, vector<MinTreePrim>& cntTree); //��С������Prim

	void Dijkstra(vector<vector<int>>& adj,vector<int>& dis,int v,int e);
	int networkDelayTime(vector<vector<int>>& times, int N, int K);//Dijkstra�㷨

	//���ֲ���
	//���Ҿ���ֵ
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
	//������߽�
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
	//�����ұ߽�
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
	bool row[10][10], col[10][10], box[10][10];//��ά������м�¼boardһ�л�һ�л�һ����
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
	λ���㳣����ʹ�÷��� �롰&������|�������^��,�ǡ�~��
	1.n&(n-1)	��ʾ��n�Ķ��������ұߵ�1��Ϊ0
	2.n&(-n)	��ʾn��n�Ĳ������������,����n�Ķ��������ұߵ�1
	3."^"	�������,��ֵͬ����λ0
	4.">>","<<"		���ƺ�����  �ֱ��ʾn�Ķ���������/���Ƽ�λ,�������1λ�൱�� n��2/��2

	��λ�����У�a^b��ʾ�޷���λ����ӣ�a&b������Ϊ�ͱ�ʾ��λ
	*/
};

