#include "pch.h"
#include "CAlgorithmclass.h"


CAlgorithmclass::CAlgorithmclass()
{
}


CAlgorithmclass::~CAlgorithmclass()
{
}

vector<vector<int>> CAlgorithmclass::threeSum(vector<int>& nums)
{
	int sSize = nums.size();
	vector<vector<int>> result;
	if (nums.empty() || sSize < 3) return vector<vector<int>>();
	sort(nums.begin(), nums.end());
	for (int i = 0; i < sSize; ++i)
	{
		if (nums[i] > 0) break;
		if (nums[i] == nums[i + 1]) continue;
		int L(i + 1), R(sSize - 1);

		while (L < R)
		{
			//if (nums[L] == nums[L + 1]) ++L;
			//if (nums[R] == nums[R - 1]) --R;
			int sum = nums[i] + nums[L] + nums[R];
			if (sum == 0)
			{
				int trip[3] = { nums[i],nums[L],nums[R] };
				vector<int> trips(trip, trip + 3);
				result.push_back(trips);
				while (nums[L] == nums[L + 1])
					++L;
				while (nums[R] == nums[R - 1])
					--R;


				++L;
				--R;
			}
			else if (sum < 0) ++L;
			else if (sum > 0) --R;
		}
	}

	return result;
}

int CAlgorithmclass::mySqrt(int x)
{
	if (x == 0)
		return 0;
	int l = 1, r = x / 2, ans = 0;
	while (l <= r)
	{
		int mid = l + (r - l) / 2;
		if ((long long)mid*mid < x) {
			l = mid + 1;
			ans = mid;
		}
		else
			r = mid - 1;
	}
	return l;
	//牛顿迭代法
	long long x0 = x;
	while (x0*x0 > x)
	{
		x0 = (x0 + x / x0) / 2;
	}
	return x0;
}

int CAlgorithmclass::climbStairs(int n)
{
	if (n <= 3)
		return n;
	n = n - 3;
	int i = 3, j = 2;
	int res = 0;
	while (n--)
	{
		res = i + j;
		j = i;
		i = res;
	}
	return res;
}

string CAlgorithmclass::simplifyPath(string path)
{
	//通过240测试例
	/*int sSize = path.size();
	string res = "";
	stack<char> pstr;
	for (int i = 0; i < sSize; ++i){
		if (pstr.empty()){
			pstr.push(path[i]);
			continue;
		}
		if (path[i] == '/'&&path[i] == pstr.top()) continue;
		if (path[i] == '.'){
			if (i+1<sSize&&path[i] == path[i + 1]){
				pstr.pop();
				while (!pstr.empty()&&pstr.top() != '/')
					pstr.pop();
				++i;
			}
			else pstr.pop();
			continue;
		}
		pstr.push(path[i]);
	}
	if (pstr.empty()) pstr.push('/');
	if (pstr.size() == 1) return "/";
	if (pstr.top() == '/') pstr.pop();
	while (!pstr.empty()){
		res = pstr.top() + res;
		pstr.pop();
	}
	return res;*/
	//也是利用栈stack
	path += "/";
	string res = "";
	stack<string> pstr;
	string temp = "";
	for (auto c : path)
	{
		if (c == '/')
		{
			if (temp == ".." && !pstr.empty())
			{
				pstr.pop();
			}
			else if (temp != "."&&temp != ".." && !temp.empty())
			{
				pstr.push(temp);
			}
			temp.clear();
		}
		else
			temp += c;
	}
	while (!pstr.empty()) {
		auto t = pstr.top();
		pstr.pop();
		res += string(t.rbegin(), t.rend()) + "/";
	}
	reverse(res.begin(), res.end());
	if (res.empty()) res = "/";
	return res;
}

vector<string> CAlgorithmclass::fullJustify(vector<string>& words, int maxWidth)
{
	int i = 0;
	vector<string> res;

	while (i < words.size())
	{
		//bool _or = false;
		int L = i;
		int R = L;
		int sSize = words[i].size();// , j = 0;
		int wordLength = sSize;
		string temp = "";
		vector<string> kongs;
		while (sSize < maxWidth&&i < words.size() - 1)
		{
			++sSize;
			sSize += words[++i].size();
			wordLength += words[i].size();
			//++j;

		}
		R = i - 1;
		if (L != R)
		{
			int shang, yushu;
			if (i == words.size() - 1)
			{
				shang = (maxWidth - (wordLength - words[i].size())) / (R - L);
				yushu = (maxWidth - (wordLength - words[i].size())) % (R - L);
			}
			else
			{
				shang = (maxWidth - (wordLength - words[i].size())) / (R - L);
				yushu = (maxWidth - (wordLength - words[i].size())) % (R - L);
			}
			int j = R - L;
			while (j--)
			{
				string kong(shang, ' ');
				kongs.push_back(kong);
			}
			if (yushu)
			{
				int k = 0;
				while (yushu--)
					kongs[k++] += ' ';
			}
			for (int n = 0; n < kongs.size(); n++, L++)
			{
				temp = temp + words[L] + kongs[n];
			}
			temp += words[R];
			res.push_back(temp);
		}
		//else if(j == 1)
		//{
		//	string kong((maxWidth - words[i-1].size()), ' ');
		//	temp = words[i-1] + kong;
		//	res.push_back(temp);
		//	//++i;
		//}
		else
		{
			string kong((maxWidth - wordLength), ' ');
			temp = words[i] + kong;
			res.push_back(temp);
			++i;
		}
	}

	return res;
}

int CAlgorithmclass::minDistance(string word1, string word2)
{
	int n = word1.size(), m = word2.size();
	if (n*m == 0)
		return n + m;
	vector<vector<int>> dp;
	vector<int> temp(m + 1);
	for (int i = 0; i < n + 1; ++i)
		dp.push_back(temp);
	for (int i = 0; i < n + 1; ++i)
		dp[i][0] = i;
	for (int j = 0; j < m + 1; ++j)
		dp[0][j] = j;
	for (int i = 1; i < n + 1; ++i)
	{
		for (int j = 1; j < m + 1; j++)
		{
			if (word1[i - 1] != word2[j - 1])
			{
				int a = dp[i - 1][j] > dp[i][j - 1] ? dp[i][j - 1] : dp[i - 1][j];
				dp[i][j] = 1 + (a > dp[i - 1][j - 1] ? dp[i - 1][j - 1] : a);
			}
			else
				dp[i][j] = dp[i - 1][j - 1];
		}
	}
	return dp[n][m];
}

void CAlgorithmclass::setZeroes(vector<vector<int>>& matrix)
{
	int n = matrix.size(), m = matrix[0].size();
	unordered_map<int, int> amap;
	vector<std::pair<int, int>> apairs;
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			if (matrix[i][j] == 0)
			{
				//amap[i] = j;
				apairs.push_back(std::make_pair(i, j));
			}
		}
	}
	for (auto & p : apairs)
	{
		for (int i = 0; i < n; ++i)
			matrix[i][p.second] = 0;
		for (int j = 0; j < m; ++j)
			matrix[p.first][j] = 0;
	}
}

bool CAlgorithmclass::searchMatrix(vector<vector<int>>& matrix, int target)
{
	int n = matrix.size(), m = matrix[0].size();
	int row = 0;
	int L = 0, R = n - 1;
	if (n > 1)
	{
		while (L <= R)
		{
			int mid = (L + R) / 2;
			if (matrix[mid][0] < target)
				L = mid;
			else if (matrix[mid][0] > target)
				R = mid;
			else if (matrix[mid][0] == target)
				return true;
			if (R - L <= 1)
				break;
		}
		if (matrix[R][0] <= target && target <= matrix[R][m - 1])
			row = R;
		else if (matrix[L][0] <= target && target <= matrix[L][m - 1])
			row = L;
		else
			return false;
	}
	L = 0, R = m - 1;
	while (L < R)
	{
		int mid = (L + R) / 2;
		if (matrix[row][mid] < target)
			L = mid;
		else if (matrix[row][mid] > target)
			R = mid;
		else if (matrix[row][mid] == target)
			return true;
		if (R - L == 1)
			return false;
	}
	return false;
}

void CAlgorithmclass::sortColors(vector<int>& nums)
{
	std::sort(nums.begin(), nums.end());
}

string CAlgorithmclass::minWindow(string s, string t)
{
	if (t.size() == 0 || s.size() == 0) return "";
	unordered_map<char, int> amap;
	for (auto& ch : t)
		amap[ch]++;
	string res = "";
	int l = 0, count = 0, ressize = INT_MAX;
	for (int i = 0; i < s.size(); ++i) {
		if (amap.find(s[i]) != amap.end()) {
			amap[s[i]]--;
			if (amap[s[i]] >= 0)
				count++;
		}
		while (count == t.size()) {
			if (ressize > i - l + 1) {
				ressize = i - l + 1;
				res = s.substr(l, i - l + 1);
			}
			if (amap.find(s[l]) != amap.end()) {
				if (amap[s[l]] >= 0)
					--count;
				amap[s[l]]++;
			}
			++l;
		}
	}
	return res;
	//滑动窗口
	/*int count[256] = { 0 };
	for (auto c : t) ++count[c];
	int len = 0, minLength = s.length();
	string res;
	for (int l = 0, r = 0; r < s.length(); ++r) {
		count[s[r]]--;
		if (count[s[r]] >= 0)
			++len;
		while (len == t.length()) {
			if (r - l + 1 <= minLength) {
				minLength = r - l + 1;
				res = s.substr(l, r - l + 1);
			}
			count[s[l]]++;
			if (count[s[l]] > 0)
				--len;
			++l;
		}
	}
	return res;*/
}

vector<vector<int>> CAlgorithmclass::combine(int n, int k)
{
	int a = 1;
	//vector<vector<int>> res;
	vector<int> temp;
	backCombine(n, k, temp, a);
	return m_intVtVt;
}

vector<vector<int>> CAlgorithmclass::subsets(vector<int>& nums)
{
	vector<int> temp;
	backSubsets(nums, temp, 0);
	return m_intVtVt;
}

bool CAlgorithmclass::exist(vector<vector<char>>& board, string word)
{
	//超出内存限制
	/*int n = board.size(), m = board[0].size();
	int k = 0;
	m_bool = false;
	vector<int> temp(m, 0);
	vector<vector<int>> _or(n,temp);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if (board[i][j] == word[0])
			{
				vector<vector<int>> _orbool = _or;
				_orbool[i][j] = 1;
				backExist(board,_orbool, word, i, j, n, m, 1);
				if (m_bool)
				{
					return true;
				}
			}
		}
	}
	return false;*/
	////////////////
	if (board.empty() || word.empty()) {
		return false;
	}
	int row = board.size(), col = board[0].size();
	vector<vector<int>> f(row, vector<int>(col, 0));
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			if (dfs(board, word, 0, i, j, f)) {
				return true;
			}
		}
	}
	return false;
}

vector<string> CAlgorithmclass::findWords(vector<vector<char>>& board, vector<string>& words)
{
	/*if (board.empty() || words.empty()) return{};
	m_row = board.size();
	m_col = board[0].size();
	Trie* root = new Trie();
	for (int k = 0; k < words.size(); ++k)
	{
		root->insert2(words[k]);
	}
	for (int i = 0; i < m_row; ++i)
	{
		for (int j = 0; j < m_col; ++j)
		{
			back_dfs(board, i, j, root);
		}
	}
	return m_strVt;*/
	return m_strVt;
}

int CAlgorithmclass::removeDuplicates(vector<int>& nums)
{
	int sSize = nums.size();
	int L = 1, R = L;
	int count = 1;
	while (R < sSize) {
		if (nums[R] == nums[R - 1]) {
			count++;
			if (count > 2)
				R++;
			else {
				nums[L] = nums[R];
				R++;
				L++;
			}
		}
		else {
			count = 1;
			nums[L] = nums[R];
			R++;
			L++;
		}
	}
	return L;
}

bool CAlgorithmclass::search(vector<int>& nums, int target)
{
	int sSize = nums.size();
	if (sSize == 0) return false;
	int start = 0, end = sSize - 1;
	int mid;
	while (start <= end)
	{
		mid = start + (end - start) / 2;
		if (nums[mid] == target)
			return true;
		if (nums[mid] == nums[start])
		{
			start++;
			continue;
		}
		if (nums[mid] > nums[start])
		{
			if (nums[mid] > target&&nums[start] <= target)
				end = mid - 1;
			else
				start = mid + 1;
		}
		else
		{
			if (nums[mid] < target&&nums[end] >= target)
				start = mid + 1;
			else
				end = mid - 1;
		}
	}
	return false;
}

ListNode * CAlgorithmclass::deleteDuplicates(ListNode * head)
{
	if (head == nullptr) return head;
	ListNode* temp = new ListNode(0);
	temp->next = head;
	ListNode* res = temp;
	ListNode* fir = head;
	bool _or = false;
	while (fir->next != nullptr)
	{
		if (fir->next->val == fir->val)
		{
			fir = fir->next;
			_or = true;
			continue;
		}
		if (_or)
		{
			temp->next = fir->next;
			fir = fir->next;
			_or = false;
			continue;
		}
		temp = temp->next;
		fir = fir->next;
	}
	if (_or)
		temp->next = nullptr;
	return res->next;
}

ListNode * CAlgorithmclass::deleteDuplicates1(ListNode * head)
{
	ListNode* pre = head;
	while (pre&&pre->next)
	{
		if (pre->val == pre->next->val)
		{
			ListNode* del = pre->next;
			pre->next = pre->next->next;
			delete del;
		}
		else
			pre = pre->next;
	}
	return head;
}

int CAlgorithmclass::largestRectangleArea(vector<int>& heights)
{
	//暴力法
	/*int sSize = heights.size();
	int res = 0;
	for (int i = 0; i < sSize; i++)
	{
		int length = 1;
		int L = i, R = i;
		while (R<sSize-1&&heights[i]<=heights[R+1])
		{
			length++;
			R++;
		}
		while (L>0&&heights[i]<=heights[L-1])
		{
			length++;
			L--;
		}
		int temp = heights[i] * length;
		res = res > temp ? res : temp;
	}*/
	//栈
	stack<int> astack;
	astack.push(-1);
	int res = 0;
	for (int i = 0; i < heights.size(); ++i)
	{
		while (astack.top() != -1 && heights[astack.top()] >= heights[i])
		{
			int sec = astack.top();
			astack.pop();
			int temp = heights[sec] * (i - astack.top() - 1);
			res = res > temp ? res : temp;
		}
		astack.push(i);
	}
	while (astack.top() != -1)
	{
		int sec = astack.top();
		astack.pop();
		int temp = heights[sec] * (heights.size() - astack.top() - 1);
		res = res > temp ? res : temp;
	}
	return res;
}

int CAlgorithmclass::maximalRectangle(vector<vector<char>>& matrix)
{
	if (matrix.size() == 0) return 0;
	int maxArea = 0;
	/*int maxArea = 0;
	int n = matrix.size(), m = matrix[0].size();
	vector<vector<int>> width(n, vector<int>(m, 0));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if (matrix[i][j] == '1')
			{
				if (j == 0)
					width[i][j] = 1;
				else
					width[i][j] = width[i][j - 1] + 1;
			}
			else
				width[i][j] = 0;
			int minwidth = width[i][j];
			for (int k = i; k>=0; k--)
			{
				int height = i - k + 1;
				minwidth = minwidth > width[k][j] ? width[k][j] : minwidth;
				int tempArea = height*minwidth;
				maxArea = maxArea > tempArea ? maxArea : tempArea;
			}
		}
	}*/
	int n = matrix.size(), m = matrix[0].size();
	vector<int> heights(m, 0);
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			if (matrix[i][j] == '1')
				heights[j]++;
			else
				heights[j] = 0;
		}
		int tempArea = largestRectangleArea(heights);
		maxArea = maxArea > tempArea ? maxArea : tempArea;
	}
	return maxArea;
}

ListNode * CAlgorithmclass::partition(ListNode * head, int x)
{
	ListNode* fir = new ListNode(-1);
	ListNode* sec = new ListNode(-1);
	auto one = fir;
	auto two = sec;
	while (head)
	{
		if (head->val < x)
		{
			fir->next = head;
			fir = fir->next;
		}
		else
		{
			sec->next = head;
			sec = sec->next;
		}
		head = head->next;
	}
	fir->next = two->next;
	sec->next = nullptr;
	return one->next;
}

bool CAlgorithmclass::isScramble(string s1, string s2)
{

	return false;
}

void CAlgorithmclass::merge(vector<int>& nums1, int m, vector<int>& nums2, int n)
{
	if (n == 0) return;
	vector<int> tempnums;
	tempnums.insert(tempnums.begin(), nums1.begin(), nums1.begin() + m);
	int i = 0, j = 0, k = 0;
	while (i < m&&j < n)
	{
		if (tempnums[i] <= nums2[j])
		{
			nums1[k++] = tempnums[i++];
		}
		else
			nums1[k++] = nums2[j++];
	}
	if (i >= m)
	{
		while (j < n)
		{
			nums1[k++] = nums2[j++];
		}
	}
	if (j >= n)
	{
		while (i < m)
		{
			nums1[k++] = tempnums[i++];
		}
	}
}

void CAlgorithmclass::solveSudoku(vector<vector<char>>& board)
{
	memset(row, false, sizeof(row));
	memset(col, false, sizeof(col));
	memset(box, false, sizeof(box));

	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			if (board[i][j] == '.')
				continue;
			int index = 3 * (i / 3) + j / 3;
			int num = board[i][j] - '0';
			row[i][num] = col[j][num] = box[index][num] = true;
		}
	}
	DFS_shudu(0, 0, board);
}

vector<int> CAlgorithmclass::grayCode(int n)
{
	int head = 1;
	vector<int> res;
	res.push_back(0);
	for (int i = 0; i < n; i++)
	{
		for (int j = res.size() - 1; j >= 0; j--)
			res.push_back(head + res[j]);
		head <<= 1;
	}
	return res;
}

vector<vector<int>> CAlgorithmclass::subsetsWithDup(vector<int>& nums)
{
	sort(nums.begin(), nums.end());
	vector<int> temp;
	backsubsetsWithDup(nums, temp, 0);
	return m_intVtVt;
}

int CAlgorithmclass::numDecodings(string s)
{
	//最快速度
	if (s[0] == 48)
		return 0;
	int cur = 1, pre = 1;
	for (int i = 1; i < s.size(); i++)
	{
		int temp = cur;
		if (s[i] == 48)
		{
			if (s[i - 1] == 49 || s[i - 1] == 50)
				cur = pre;
			else
				return 0;
		}
		else if (s[i - 1] == 49 || (s[i - 1] == 50 && 48 < s[i] && s[i] < 55))
			cur = cur + pre;
		pre = temp;
	}
	return cur;
}

ListNode * CAlgorithmclass::reverseBetween(ListNode * head, int m, int n)
{
	ListNode* fir = new ListNode(-1);
	fir->next = head;
	ListNode* res = fir;
	int temp = n - m;
	while (--m)
	{
		fir = fir->next;
		head = head->next;
	}
	while (temp--)
	{
		ListNode* sec = head->next->next;
		head->next->next = fir->next;
		fir->next = head->next;
		head->next = sec;
	}
	return res->next;
}

vector<string> CAlgorithmclass::restoreIpAddresses(string s)
{
	//递归
	/*string segment;
	back_IpAddresses(s, 0, segment);
	return m_strVt;*/
	//有效的暴力解法
	string temp;
	for (int a = 1; a < 4; ++a)
	{
		for (int b = 1; b < 4; ++b)
		{
			for (int c = 1; c < 4; ++c)
			{
				for (int d = 1; d < 4; ++d)
				{
					if (a + b + c + d == s.size())
					{
						int n1 = stoi(s.substr(0, a));
						int n2 = stoi(s.substr(a, b));
						int n3 = stoi(s.substr(a + b, c));
						int n4 = stoi(s.substr(a + b + c, d));
						if (n1 < 256 && n2 < 256 && n3 < 256 && n4 < 256)
						{
							temp = temp + to_string(n1) + "." + to_string(n2)
								+ "." + to_string(n3) + "." + to_string(n4);
							if (temp.size() == s.size() + 3)
								m_strVt.push_back(temp);
							temp = "";
						}
					}
				}
			}
		}
	}
	return m_strVt;
}

vector<int> CAlgorithmclass::preorderTraversal(TreeNode * root)
{
	//迭代方法
	/*if (!root) return{};
	vector<int> ans;
	stack<TreeNode*> s;
	s.push(root);
	while (!s.empty())
	{
		TreeNode* temp = s.top();
		s.pop();
		ans.push_back(temp->val);
		if (temp->right) s.push(temp->right);
		if (temp->left) s.push(temp->left);
	}
	return ans;*/
	//迭代优化，只压入右节点
	if (!root) return{};
	vector<int> ans;
	stack<TreeNode*> s;
	while (true)
	{
		if (root) {
			ans.push_back(root->val);
			if (root->right)
				s.push(root->right);
			root = root->left;
		}
		else if (s.empty())
			return ans;
		else {
			root = s.top();
			s.pop();
		}
	}
}

vector<int> CAlgorithmclass::inorderTraversal(TreeNode * root)
{
	//递归或者利用栈->进行迭代
	/*if (root == NULL)
		return{};
	stack<TreeNode*> s;
	while (!s.empty()||root)
	{
		if (root)
		{
			s.push(root);
			root = root->left;
		}
		else
		{
			root = s.top();
			s.pop();
			m_intVt.push_back(root->val);
			root = root->right;
		}
	}
	return m_intVt;*/
	//颜色记忆
	int white = 0, gray = 1;
	stack<std::pair<int, TreeNode*>> s;
	s.push(make_pair(white, root));
	while (!s.empty())
	{
		int cloor = s.top().first;
		TreeNode* temp = s.top().second;
		s.pop();
		if (temp == NULL)
			continue;
		if (cloor == white)
		{
			s.push(make_pair(white, temp->right));
			s.push(make_pair(gray, temp));
			s.push(make_pair(white, temp->left));
		}
		else
			m_intVt.push_back(temp->val);
	}
	return m_intVt;
}

vector<int> CAlgorithmclass::postorderTraversal(TreeNode * root)
{
	//迭代
	/*if (!root) return{};
	deque<int> d;
	stack<TreeNode*> s;
	s.push(root);
	while (!s.empty())
	{
		TreeNode* temp = s.top();
		s.pop();
		d.push_front(temp->val);
		if (temp->left) s.push(temp->left);
		if (temp->right) s.push(temp->right);
	}
	return vector<int>(d.begin(),d.end());*/
	//不依靠队列反转
	if (!root) return{};
	vector<int> ans;
	stack<TreeNode*> s;
	s.push(root);
	while (!s.empty())
	{
		TreeNode* temp = s.top();
		if (!temp) {
			s.pop();
			ans.push_back(s.top()->val);
			s.pop();
			continue;
		}
		s.push(nullptr);
		if (temp->right) s.push(temp->right);
		if (temp->left) s.push(temp->left);
	}
	return ans;
}

vector<TreeNode*> CAlgorithmclass::generateTrees(int n)
{
	vector<TreeNode*> ans;
	if (n == 0) return ans;
	ans = back_generateTrees(1, n);
	return ans;
}

int CAlgorithmclass::numTrees(int n)
{
	//递归方法
	/*int ans;
	if (n == 0) return 0;
	ans = back_numTrees(1, n);
	return ans;*/
	//动态规划 运用迭代公式
	//vector<int> G(n+1,0);
	int *dp = new int[n + 1];
	for (int i = 0; i <= n; ++i) dp[i] = 0;
	dp[0] = 1;
	dp[1] = 1;
	for (int i = 2; i <= n; ++i)
	{
		for (int j = 1; j <= i; ++j)
		{
			dp[i] += dp[j - 1] * dp[i - j];
		}
	}
	int ans = dp[n];
	delete[] dp;
	return ans;
}

bool CAlgorithmclass::isInterleave(string s1, string s2, string s3)
{
	//递归
	/*if ((s1.size() + s2.size()) != s3.size())
		return false;
	return back_inInterleave(s1,s2,s3,0,0,0);*/
	//动态规划
	if ((s1.size() + s2.size()) != s3.size())
		return false;
	vector<vector<bool>> dp(s1.size() + 1, vector<bool>(s2.size() + 1, false));
	dp[0][0] = true;
	for (int i = 0; i <= s1.size(); ++i) {
		for (int j = 0; j <= s2.size(); ++j) {
			if (i > 0 && s1[i - 1] == s3[i + j - 1]) {
				dp[i][j] = dp[i][j] || dp[i - 1][j];
			}
			if (j > 0 && s2[j - 1] == s3[i + j - 1]) {
				dp[i][j] = dp[i][j] || dp[i][j - 1];
			}
		}
	}
	return dp[s1.size()][s2.size()];
}

bool CAlgorithmclass::isValidBST(TreeNode * root)
{
	//迭代
	/*TreeNode* temp = NULL;
	stack<TreeNode*> s;
	while (!s.empty()||root)
	{
		if (root)
		{
			s.push(root);
			root = root->left;
		}
		else
		{
			root = s.top();
			s.pop();
			if (temp&&temp->val >= root->val)
				return false;
			temp = root;
			root->right;
		}
	}
	return true;*/
	//递归
	if (root == NULL) return true;
	if (!isValidBST(root->left))
		return false;
	if (m_tree&&m_tree->val >= root->val)
		return false;
	m_tree = root;
	if (!isValidBST(root->right))
		return false;
	return true;
}

bool CAlgorithmclass::isSameTree(TreeNode * p, TreeNode * q)
{
	if (p == NULL && q == NULL)
		return true;
	if (p == NULL || q == NULL)
		return false;
	if (p->val != q->val)
		return false;
	if (isSameTree(p->left, q->left) && isSameTree(p->right, q->right))
		return true;
	return false;
	//return isSameTree(p->left,q->left)&&isSameTree(p->right,q->right);
}

void CAlgorithmclass::recoverTree(TreeNode * root)
{
	TreeNode* fir = NULL;
	TreeNode* one = NULL;
	TreeNode* two = NULL;
	stack<TreeNode*> s;
	while (!s.empty() || root) {
		if (root) {
			s.push(root);
			root = root->left;
		}
		else {
			root = s.top();
			s.pop();
			if (fir && fir->val >= root->val) {
				if (one == NULL) {
					one = fir;
					two = root;
				}
				else two = root;
			}
			fir = root;
			root = root->right;
		}
	}
	int temp = one->val;
	one->val = two->val;
	two->val = temp;
}

bool CAlgorithmclass::isSymmetric(TreeNode * root)
{
	//递归
	/*if (root == NULL) return true;
	TreeNode* lefttree = root->left;
	TreeNode* righttree = root->right;
	return back_isSymmetric(lefttree,righttree);*/
	//迭代
	if (root == NULL) return true;
	queue<TreeNode*> q;
	q.push(root);
	q.push(root);
	while (!q.empty())
	{
		TreeNode* fir = q.front();
		q.pop();
		TreeNode* sec = q.front();
		q.pop();
		if (fir == NULL && sec == NULL) continue;
		if (fir == NULL || sec == NULL) return false;
		if (fir->val != sec->val) return false;
		q.push(fir->left);
		q.push(sec->right);
		q.push(fir->right);
		q.push(sec->left);
	}
	return true;
}

vector<vector<int>> CAlgorithmclass::levelOrder(TreeNode * root)
{
	if (root == NULL) return{};
	//第一种迭代方法
	/*queue<std::pair<int,TreeNode*>> q;
	int level = 1;
	q.push(make_pair(1,root));
	vector<int> temp;
	while (!q.empty())
	{
		TreeNode* fir = q.front().second;
		int a = q.front().first;
		if (a == level)
			temp.push_back(fir->val);
		else
		{
			level++;
			m_intVtVt.push_back(temp);
			temp.clear();
			temp.push_back(fir->val);
		}
		q.pop();
		if (fir->left)
			q.push(make_pair(a+1,fir->left));
		if (fir->right)
			q.push(make_pair(a+1,fir->right));
	}
	m_intVtVt.push_back(temp);
	return m_intVtVt;*/
	//第二种迭代运算
	queue<TreeNode*> Q;
	TreeNode* p;
	Q.push(root);
	while (Q.empty() == 0) {
		vector<int> a;
		int width = Q.size();
		for (int i = 0; i < width; i++) {
			p = Q.front();
			a.push_back(p->val);
			Q.pop();
			if (p->left) Q.push(p->left);
			if (p->right) Q.push(p->right);
		}
		m_intVtVt.push_back(a);
	}
	return m_intVtVt;
}

vector<vector<int>> CAlgorithmclass::zigzagLevelOrder(TreeNode * root)
{
	if (root == NULL) return{};
	queue<TreeNode*> Q;
	TreeNode* p;
	Q.push(root);
	bool reverse(true);
	while (Q.empty() == 0) {
		reverse = !reverse;
		deque<int> d;
		int width = Q.size();
		for (int i = 0; i < width; i++) {
			p = Q.front();
			if (reverse)
				d.push_front(p->val);
			else
				d.push_back(p->val);
			Q.pop();
			if (p->left) Q.push(p->left);
			if (p->right) Q.push(p->right);
		}
		m_intVtVt.push_back(vector<int>(d.begin(), d.end()));
	}
	return m_intVtVt;
}

int CAlgorithmclass::maxDepth(TreeNode * root)
{
	//递归方式 深度优先
	/*if (root == NULL) return 0;
	back_maxDepth(root, 0);
	return m_int;*/
	//迭代方法 广度优先
	if (root == NULL) return 0;
	queue<TreeNode*> q;
	q.push(root);
	int deep = 0;
	while (!q.empty())
	{
		deep++;
		int num = q.size();
		for (int i = 0; i < num; ++i)
		{
			TreeNode* temp = q.front();
			q.pop();
			if (temp->left) q.push(temp->left);
			if (temp->right) q.push(temp->right);
		}
	}
	return deep;
}

TreeNode * CAlgorithmclass::buildTree(vector<int>& preorder, vector<int>& inorder)
{
	if (preorder.size() == 0) return nullptr;
	m_int = 0;
	for (int i = 0; i < inorder.size(); i++)
		m_map[inorder[i]] = i;
	TreeNode* root = back_buildTree(preorder, 0, inorder.size());
	return root;
}

TreeNode * CAlgorithmclass::buildTree1(vector<int>& inorder, vector<int>& postorder)
{
	if (postorder.size() == 0) return nullptr;
	m_int = postorder.size() - 1;
	m_intVt = postorder;
	for (int i = 0; i < inorder.size(); i++)
		m_map[inorder[i]] = i;
	TreeNode* root = back_buildTree1(0, inorder.size());
	return root;
}

vector<vector<int>> CAlgorithmclass::levelOrderBottom(TreeNode * root)
{
	//迭代
	/*if (root == NULL) return{};
	deque<vector<int>> d;
	queue<TreeNode*> q;
	q.push(root);
	while (!q.empty())
	{
		int i = q.size();
		vector<int> lev(i);
		for (int j = 0; j < i; ++j)
		{
			TreeNode* temp = q.front();
			q.pop();
			lev[j] = temp->val;
			if(temp->left) q.push(temp->left);
			if(temp->right) q.push(temp->right);
		}
		d.push_front(lev);
	}
	return vector<vector<int>>(d.begin(),d.end());*/
	//递归
	return{};
}

TreeNode * CAlgorithmclass::sortedArrayToBST(vector<int>& nums)
{
	TreeNode* root = back_sortedArrayToBST(nums, 0, nums.size());
	return root;
}

TreeNode * CAlgorithmclass::sortedListToBST(ListNode * head)
{
	//把链表存到数组中
	/*vector<int> nums;
	while (head)
	{
		nums.push_back(head->val);
		head = head->next;
	}
	TreeNode* root = back_sortedArrayToBST(nums, 0, nums.size());
	return root;*/
	//根据中序遍历的方法
	return back_sortedListToBST(head, NULL);
}

bool CAlgorithmclass::isBalanced(TreeNode * root)
{
	int height;
	return back_isBalanced(root, height);
}

int CAlgorithmclass::minDepth(TreeNode * root)
{
	//深度递归
	/*if (root == NULL) return 0;
	return back_minDepth(root);*/
	//广度迭代
	/*if (root == NULL) return 0;
	int mindeep = 0;
	queue<TreeNode*> q;
	q.push(root);
	while (!q.empty())
	{
		++mindeep;
		int s = q.size();
		for (int i = 0; i < s; ++i)
		{
			TreeNode* temp = q.front();
			q.pop();
			if (temp->left) q.push(temp->left);
			if (temp->right) q.push(temp->right);
			if (temp->left == NULL&&temp->right == NULL) return mindeep;
		}
	}
	return mindeep;*/
	//深度迭代
	if (root == NULL) return 0;
	stack<pair<TreeNode*, int> > stack;
	stack.push(make_pair(root, 1));
	int res = INT_MAX;
	while (!stack.empty()) {
		TreeNode* node = stack.top().first;
		int depth = stack.top().second;
		stack.pop();
		if (!node->left && !node->right)
			res = min(res, depth);
		if (node->left)
			stack.push(make_pair(node->left, depth + 1));
		if (node->right)
			stack.push(make_pair(node->right, depth + 1));
	}
	return res;
}

bool CAlgorithmclass::hasPathSum(TreeNode * root, int sum)
{
	//深度递归
	/*m_int = sum;
	return back_hasPathSum(root,0);*/
	//深度迭代
	if (root == NULL) return 1;
	stack<std::pair<TreeNode*, int>> s;
	s.push(make_pair(root, root->val));
	while (!s.empty())
	{
		TreeNode* temp = s.top().first;
		int val = s.top().second;
		s.pop();
		if (temp->left == NULL && temp->right == NULL && val == sum)
			return true;
		if (temp->left)
		{
			TreeNode* l = temp->left;
			s.push(make_pair(l, val + l->val));
		}
		if (temp->right)
		{
			TreeNode* r = temp->right;
			s.push(make_pair(r, val + r->val));
		}
	}
	return false;
}

vector<vector<int>> CAlgorithmclass::pathSum(TreeNode * root, int sum)
{
	if (root == NULL) return{};
	m_int = sum;
	vector<int> temp;
	back_PathSum(root, 0, temp);
	return m_intVtVt;
}

void CAlgorithmclass::flatten(TreeNode * root)
{
	if (root == NULL) return;
	while (root)
	{
		if (root->left)
		{
			TreeNode* temp = root->left;
			while (temp->right != NULL)
				temp = temp->right;
			temp->right = root->right;
			root->right = root->left;
			root->left = nullptr;
		}
		root = root->right;
	}
	//递归
	/*if (!root)
		return;
	flatten(root->left);
	flatten(root->right);
	TreeNode* tmp = root->left;
	if (!tmp) return;
	while (tmp->right)
		tmp = tmp->right;
	tmp->right = root->right;
	root->right = root->left;
	root->left = nullptr;*/
}

int CAlgorithmclass::numDistinct(string s, string t)
{
	//递归超时
	/*m_int = 0;
	back_numDistinct(s, t, 0, 0);
	return m_int;*/
	//动态规划二维
	/*vector<vector<long>> dp(t.size()+1,vector<long>(s.size()+1,0));
	for (int i = 0; i < s.size()+1; ++i)
		dp[0][i] = 1;
	for (int i = 1; i < t.size()+1; ++i)
	{
		for (int j = i; j < s.size()+1; ++j)
		{
			if (s[j-1] == t[i-1])
				dp[i][j] = dp[i][j - 1] + dp[i - 1][j - 1];
			else
				dp[i][j] = dp[i][j - 1];
		}
	}
	return dp[t.size()][s.size()];*/
	//动态规划一维
	vector<long> dp(s.size() + 1, 1);
	for (int i = 1; i < t.size() + 1; ++i)
	{
		long last = dp[i - 1];
		dp[i - 1] = 0;
		for (int j = i; j < s.size() + 1; ++j)
		{
			long cur = dp[j];
			if (t[i - 1] == s[j - 1])
				dp[j] = dp[j - 1] + last;
			else
				dp[j] = dp[j - 1];
			last = cur;
		}
	}
	return dp[s.size()];
}

TreeNode1 * CAlgorithmclass::connect(TreeNode1 * root)
{
	if (root == NULL) return NULL;
	//非常亮 迭代
	/*queue<Node*> q;
	q.push(root);
	while (!q.empty())
	{
		vector<TreeNode1*> v;
		int s = q.size();
		for (int i = 0; i < s; ++i)
		{
			TreeNode1* temp = q.front();
			q.pop();
			if (temp->left){
				q.push(temp->left);
				v.push_back(temp->left);
			}
			if (temp->right){
				q.push(temp->right);
				v.push_back(temp->right);
			}
		}
		for (int i = 0; i < v.size()-1&&v.size(); ++i)
			v[i]->next = v[i + 1];
	}*/
	//额外常量 迭代
	/*TreeNode1* pre = root;
	TreeNode1* cur = nullptr;
	while (pre->left){
		cur = pre;
		while (cur){
			cur->left->next = cur->right;
			if (cur->next)
				cur->right->next = cur->next->left;
			cur = cur->next;
		}
		pre = pre->left;
	}*/
	//递归
	back_connect(root);
	return root;
}

TreeNode1 * CAlgorithmclass::connect2(TreeNode1 * root)
{
	if (root == NULL) return NULL;
	back_connect2(root);
	return root;
}

vector<vector<int>> CAlgorithmclass::generate(int numRows)
{
	vector<vector<int>> ans(numRows);
	for (int i = 0; i < numRows; ++i) {
		ans[i] = vector<int>(i + 1, 0);
		ans[i][0] = 1;
		ans[i][i] = 1;
	}
	if (numRows <= 2) return ans;
	for (int i = 2; i < numRows; ++i)
		for (int j = 1; j < ans[i].size() - 1; ++j) ans[i][j] = ans[i - 1][j - 1] + ans[i - 1][j];
	return ans;
}

vector<int> CAlgorithmclass::getRow(int rowIndex)
{
	if (rowIndex == 0) return{ 1 };
	if (rowIndex == 1) return{ 1,1 };
	vector<int> ans = { 1,1 };
	while (rowIndex > 1)
	{
		int pre, cur(1);
		for (int i = 1; i < ans.size(); ++i)
		{
			pre = ans[i];
			ans[i] = ans[i] + cur;
			cur = pre;
		}
		ans.push_back(1);
		rowIndex--;
	}
	return ans;
}

int CAlgorithmclass::minimumTotal(vector<vector<int>>& triangle)
{
	//自顶向下
	/*int ans(INT_MAX);
	int row = triangle.size();
	for (int i = 1; i < row; ++i){
		for (int j = 0; j < triangle[i].size(); ++j){
			if (j == 0) {
				triangle[i][j] += triangle[i - 1][j];
				continue;
			}
			if (j == triangle[i].size() - 1) {
				triangle[i][j] += triangle[i - 1][j-1];
				continue;
			}
			triangle[i][j] += triangle[i - 1][j] > triangle[i - 1][j - 1] ? triangle[i - 1][j - 1] : triangle[i - 1][j];
		}
	}
	for (int i = 0; i < triangle[row-1].size(); ++i)
		ans = ans > triangle[row - 1][i] ? triangle[row - 1][i] : ans;
	return ans;*/
	//自底向上 更简便
	int row = triangle.size();
	vector<int> dp = triangle[row - 1];
	for (int i = row - 2; i > 0; --i) {
		for (int j = 0; j < triangle[i].size(); ++j) {
			int temp = dp[j] > dp[j + 1] ? dp[j + 1] : dp[j];
			dp[j] = temp + triangle[i][j];
		}
	}
	return dp[0];
}

int CAlgorithmclass::maxProfit(vector<int>& prices)
{
	if (prices.empty()) return 0;
	int ans(0), profit(0), pre = prices[0];
	for (int i = 1; i < prices.size(); ++i) {
		if (pre > prices[i])
			pre = prices[i];
		else
			profit = prices[i] - pre;
		ans = ans > profit ? ans : profit;
	}
	return ans;
}

int CAlgorithmclass::maxProfit2(vector<int>& prices)
{
	//第一种贪心法
	if (prices.empty()) return 0;
	/*int ans(0),pre = prices[0],curprofit(0);
	for (int i = 1; i < prices.size(); ++i)
	{
		if (pre > prices[i]||prices[i] < prices[i - 1]) {
			ans += curprofit;
			curprofit = 0;
			pre = prices[i];
		}
		else {
			curprofit = prices[i] - pre;
			continue;
		}
		ans += curprofit;
	}
	ans += curprofit;*/
	//第二种贪心法
	int ans(0);
	for (int i = 1; i < prices.size(); ++i)
	{
		int temp = prices[i] - prices[i - 1];
		if (temp > 0) ans += temp;
	}
	return ans;
}

int CAlgorithmclass::maxProfit3(vector<int>& prices)
{
	if (prices.empty()) return 0;
	int a = -prices[0], b = INT_MIN, c = INT_MIN, d = INT_MIN;
	for (int i = 1; i < prices.size(); ++i)
	{
		a = max(a, -prices[i]);
		b = max(b, a + prices[i]);
		c = max(c, b - prices[i]);
		d = max(d, c + prices[i]);
	}
	return max(b, max(d, 0));
}

int CAlgorithmclass::maxProfit4(int k, vector<int>& prices)
{
	if (prices.empty()) return 0;
	if (k > prices.size() / 2) {
		int ans(0);
		for (int i = 1; i < prices.size(); ++i) {
			int temp = prices[i] - prices[i - 1];
			if (temp > 0) ans += temp;
		}
		return ans;
	}
	vector<vector<vector<int>>> dp(prices.size(), vector<vector<int>>(k + 1, vector<int>(2, 0)));
	for (int i = 0; i < prices.size(); ++i)
	{
		for (int j = k; j > 0; --j)
		{
			if (i == 0) {
				dp[0][j][0] = 0;
				dp[0][j][1] = INT_MIN;
				continue;
			}
			dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
			dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);
		}
	}
	return dp[prices.size() - 1][k][0];
}

int CAlgorithmclass::maxProfit5(vector<int>& prices)
{
	if (prices.size() == 0) return 0;
	int dp_i_1 = 0, dp_i_0 = -prices[0], dp_temp = 0;
	for (int i = 1; i < prices.size(); ++i)
	{
		int temp = dp_i_1;
		dp_i_0 = max(dp_i_0, dp_temp - prices[i]);
		dp_i_1 = max(dp_i_1, dp_i_0 + prices[i]);
		dp_temp = temp;
	}
	return dp_i_1;
}

int CAlgorithmclass::maxPathSum(TreeNode * root)
{
	m_int = root->val;
	back_maxPathSum(root);
	return m_int;
}

bool CAlgorithmclass::isPalindrome(string s)
{
	int l(0), r(s.size() - 1);
	while (l < r) {
		while (l < r && (s[l] < 48 || (57 < s[l] && s[l] < 65) || (90 < s[l] && s[l] < 97) || 122 < s[l])) l++;
		while (l < r && (s[r] < 48 || (57 < s[r] && s[r] < 65) || (90 < s[r] && s[r] < 97) || 122 < s[r])) r--;
		if (s[l] == s[r] || s[l] > 57 && s[r] > 57 && (s[l] + 32 == s[r] || s[l] - 32 == s[r])) {
			l++;
			r--;
		}
		else  return false;
	}
	return true;
}

void CAlgorithmclass::maxword(string& str) {
	vector<std::pair<string, int>> ans;
	int start = 0;
	for (int i = 0; i < str.size(); ++i)
	{
		//int start = i;
		if (str[i] != ' ')
			continue;
		string temp = str.substr(start, i - start);
		start = i + 1;
		int _or(true);
		for (int j = 0; j < ans.size(); ++j)
		{
			if (temp == ans[j].first)
			{
				ans[j].second += 1;
				_or = false;
				break;
			}
		}
		if (_or)
			ans.push_back(make_pair(temp, 1));
	}
	int res(0), y;
	for (int j = 0; j < ans.size(); ++j)
	{
		if (ans[j].second > ans[res].second)
			res = j;
	}
	start = 0;
	for (int i = 0; i < str.size(); ++i)
	{
		if (str[i] != ' ')
			continue;
		string temp = str.substr(start, i - start);
		if (temp == ans[res].first) {
			y = start;
			break;
		}
		start = i + 1;
	}
	//m_two.s = ans[res].first;//返回出现次数最多的单词
	//m_two.y = y;//返回第一次出现的位置
}

int CAlgorithmclass::ladderLength(string beginWord, string endWord, vector<string>& wordList)
{
	//双向广度搜索，从小的队列开始搜索
	//遍历所有候选单词，并判断是否可以转换即使用isconvert()
	/*queue<string> q;
	vector<bool> visited(wordList.size(),false);
	queue<string> q2;
	vector<bool> visited2(wordList.size(), false);
	int idx(-1),idx2(-1);
	for (int i = 0; i < wordList.size(); ++i) {
		if (wordList[i] == beginWord) idx = i;
		if (wordList[i] == endWord) idx2 = i;
	}
	if (idx != -1) visited[idx] = true;
	if (idx2 != -1) visited2[idx2] = true;
	else return 0;
	q.push(beginWord);
	q2.push(endWord);
	int count(0);
	while (!q.empty()&&!q2.empty())
	{
		count++;
		if (q.size() > q2.size()) {
			swap(q, q2);
			swap(visited, visited2);
		}
		int sSize = q.size();
		for (int i = 0; i < sSize; ++i)
		{
			string s = q.front();
			q.pop();
			for (int j = 0; j < wordList.size(); ++j)
			{
				if (visited[j])
					continue;
				if (!isconvert(s, wordList[j]))
					continue;
				if (visited2[j])
					return count + 1;
				visited[j] = true;
				q.push(wordList[j]);
			}
		}
	}
	return 0;*/
	//也是广度搜索
	//因为单词是由a~z这有限数量的字符组成的，可以遍历当前单词能转换成的所有单词，
	//判断其是否包含在候选单词中。候选单词用HashSet保存，可以大大提高判断包含关系的性能
	queue<string> q;
	queue<string> q2;
	unordered_set<string> dict(wordList.begin(), wordList.end());
	if (dict.find(endWord) == dict.end()) return 0;
	unordered_set<string> aset;
	unordered_set<string> aset2;
	aset.insert(beginWord);
	aset2.insert(endWord);
	q.push(beginWord);
	q2.push(endWord);
	int count(0);
	while (!q.empty() && !q2.empty())
	{
		count++;
		if (q.size() > q2.size()) {
			swap(q, q2);
			swap(aset, aset2);
		}
		int sSize = q.size();
		for (int i = 0; i < sSize; ++i)
		{
			string s = q.front();
			q.pop();
			for (int j = 0; j < s.size(); ++j)
			{
				char cj = s[j];
				for (char k = 'a'; k <= 'z'; ++k)
				{
					s[j] = k;
					if (aset.find(s) != aset.end())
						continue;
					if (aset2.find(s) != aset2.end())
						return count + 1;
					if (dict.find(s) != dict.end()) {
						aset.insert(s);
						q.push(s);
					}
				}
				s[j] = cj;
			}
		}
	}
	return 0;
}

vector<vector<string>> CAlgorithmclass::findLadders(string beginWord, string endWord, vector<string>& wordList)
{
	//采用单向广度搜索
	/*unordered_set<string> dict(wordList.begin(), wordList.end());
	if (dict.find(endWord) == dict.end()) return{};
	vector<vector<string>> ans;
	unordered_set<string> visited;
	queue<vector<string>> q;
	vector<string> strVt;
	strVt.push_back(beginWord);
	q.push(strVt);
	visited.insert(beginWord);
	bool isend = false;
	while (!q.empty()&&!isend)
	{
		int sSize = q.size();
		unordered_set<string> subvisited;
		for (int i = 0; i < sSize; ++i)
		{
			vector<string> path = q.front();
			q.pop();
			string s = path[path.size() - 1];
			for (int m = 0; m < s.size(); ++m)
			{
				char cm = s[m];
				for (char k = 'a'; k <= 'z'; ++k)
				{
					s[m] = k;
					if (dict.find(s) == dict.end())
						continue;
					if (visited.find(s) != visited.end())
						continue;
					if (s == endWord)
					{
						isend = true;
						path.push_back(s);
						ans.push_back(path);
						path.pop_back();
					}
					path.push_back(s);
					q.push(path);
					path.pop_back();
					subvisited.insert(s);
				}
				s[m] = cm;
			}

		}
		visited.insert(subvisited.begin(), subvisited.end());
	}

	return ans;*/
	return {};
}

int CAlgorithmclass::longestConsecutive(vector<int>& nums)
{
	//暴力法
	/*int ans(0);
	sort(nums.begin(), nums.end());
	for (int i = 0; i < nums.size(); ++i)
	{
		int temp(1),num = nums[i];
		for (int j = i+1; j < nums.size(); ++j)
		{
			if ( num== nums[j] - 1) {
				num = nums[j];
				temp++;
			}
		}
		ans = max(ans, temp);
	}
	return ans;*/
	//使用map
	if (nums.empty()) return 0;
	map<int, int> amap;
	int pre, ans(0), count(1);
	for (int i = 0; i < nums.size(); ++i) amap[nums[i]]++;
	map<int, int>::iterator it = amap.begin();
	pre = it++->first;
	for (; it != amap.end(); ++it)
	{
		if (it->first == pre + 1)
			count++;
		else {
			ans = max(ans, count);
			count = 1;
		}
		pre = it->first;
	}
	return max(ans, count);
}

int CAlgorithmclass::sumNumbers(TreeNode * root)
{
	back_sumNumbers(root, 0);
	return m_int;
}

void CAlgorithmclass::yasuosuanfa(string& s)
{
	int i(0);
	while (i < s.size())
	{
		if (s[i] == ']') {
			int j = i;
			int k(0);
			while (s[j] != '[')
			{
				if (s[j] == '|')
					k = j;
				j--;
			}
			int chong = stoi(s.substr(j + 1, k - j - 1));
			string s1 = s.substr(k + 1, i - k - 1);
			while (--chong)
				s1 += s1;
			s = s.replace(j, i - j + 1, s1);
			i = j;
		}
		i++;
	}
}

vector<int> CAlgorithmclass::kandaodelou(vector<int>& x)
{
	vector<int> a, b;
	stack<int> s1, s2;
	for (int i = 0; i < x.size(); i++)
	{
		a.push_back(s1.size());
		while (!s1.empty() && s1.top() <= x[i])
			s1.pop();
		s1.push(x[i]);
	}
	for (int i = x.size() - 1; i >= 0; i--)
	{
		b.push_back(s2.size());
		while (!s2.empty() && s2.top() <= x[i])
			s2.pop();
		s2.push(x[i]);
	}
	reverse(b.begin(), b.end());
	for (int i = 0; i < a.size(); i++)
		a[i] = a[i] + b[i] + 1;
	return a;
}

int CAlgorithmclass::xiuxi(vector<int>& gs, vector<bool>& jsf, int n)
{
	vector<vector<int>> dp(3, vector<int>(n + 1));
	dp[0][0] = dp[1][0] = dp[2][0] = 0;
	for (int i = 1; i <= n; i++)
	{
		if (jsf[i - 1])
			dp[1][i] = min(dp[0][i - 1], dp[2][i - 1]);
		if (gs[i - 1])
			dp[2][i] = min(dp[0][i - 1], dp[1][i - 1]);
		dp[0][i] = min(dp[0][i - 1], min(dp[1][i - 1], dp[2][i - 1])) + 1;
	}
	return min(dp[0][n], min(dp[1][n], dp[2][n]));
}

void CAlgorithmclass::linshi(int k, vector<vector<int>>& s)
{
	//int row, int m = s[0].size();
	//int l = 0, r = s.size() - 1;
	//while (l <= r) {
	//	int mid = (l + r) / 2;
	//	if (s[mid][0]<k)l = mid;
	//	else if (s[mid][0]>k)r = mid;
	//	//else if (s[mid][0] == k) std::cout << "存在" << endl;
	//	if (r - l <= 1) break;
	//}
	//if (s[r][0] <= k&&k <= s[r][m - 1])row = r;
	//else if (s[l][0] <= k&&k <= s[l][m - 1])row = l;
	////else cout << "不存在" << endl;
	//l = 0, r = m - 1;
	//while (l<r) {
	//	int mid = (l + r) / 2;
	//	if (s[row][mid]<k)l = mid;
	//	else if (s[row][mid]>k)r = mid;
	//	//else if (s[row][mid] == k) cout << "存在" << endl;
	//	if (r - l == 1) break;
	//}
	////if (s[row][l] == k || s[row][r] == k) cout << "存在" << endl;
	////else cout << "不存在" << endl;
}

void CAlgorithmclass::solve(vector<vector<char>>& board)
{
	//速度太慢，内存击败百分百
	/*if (board.empty()) return;
	vector<tuple<int, int, bool>> m_o;
	queue<tuple<int, int, bool>> q;
	int row = board.size(), col = board[0].size();
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			if (board[i][j] == 88) continue;
			if (i == 0 || j == 0 || i == row-1 || j == col-1) {
				q.push(std::make_tuple(i, j, true));
				m_o.push_back(std::make_tuple(i, j, true));
			}
			else
				m_o.push_back(std::make_tuple(i, j, false));
		}
	}
	while (!q.empty())
	{
		int sSize = q.size();
		for (int i = 0; i < sSize; ++i)
		{
			int m = std::get<0>(q.front());
			int n = std::get<1>(q.front());
			q.pop();
			for (int j = 0; j < m_o.size(); j++)
			{
				if (std::get<2>(m_o[j]) == true) continue;
				if (((std::get<0>(m_o[j]) == m - 1|| std::get<0>(m_o[j]) == m + 1) && std::get<1>(m_o[j]) == n)||
					std::get<0>(m_o[j]) == m && (std::get<1>(m_o[j]) == n - 1|| std::get<1>(m_o[j]) == n + 1)){
					std::get<2>(m_o[j]) = true;
					q.push(m_o[j]);
				}
			}
		}
	}
	for (int i = 0; i < m_o.size(); ++i)
	{
		if (std::get<2>(m_o[i]) == true) continue;
		board[std::get<0>(m_o[i])][std::get<1>(m_o[i])] = 88;
	}*/
	//递归解法
	if (board.empty()) return;
	m_pairVt = { make_pair(0,1),make_pair(0,-1), make_pair(-1,0), make_pair(1,0) };
	int row = board.size(), col = board[0].size();
	for (int i = 0; i < row; ++i) {
		if (board[i][0] == 79) back_solve(i, 0, row, col, board);
		if (board[i][col - 1] == 79) back_solve(i, col - 1, row, col, board);
	}
	for (int i = 0; i < col; ++i) {
		if (board[0][i] == 79) back_solve(0, i, row, col, board);
		if (board[row - 1][i] == 79) back_solve(row - 1, i, row, col, board);
	}
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			if (board[i][j] == 79) board[i][j] = 88;
			if (board[i][j] == 90) board[i][j] = 79;
		}
	}
}

vector<vector<string>> CAlgorithmclass::partition(string s)
{
	if (s.size() == 0) return{};
	m_int = s.size();
	vector<vector<bool>> dp(m_int, vector<bool>(m_int, 0));
	for (int i = 0; i < m_int; ++i) {
		for (int j = 0; j <= i; ++j) {
			if (s[i] == s[j] && (i - j < 2 || dp[j + 1][i - 1]))
				dp[j][i] = true;
		}
	}
	vector<string> temp;
	back_partition(s, 0, temp, dp);
	return m_strVtVt;
}

int CAlgorithmclass::minCut(string s)
{
	//动态规划合并
	m_int = s.size();
	vector<int> dp(m_int);
	vector<vector<bool>> dp1(m_int, vector<bool>(m_int, 0));
	for (int i = 0; i < m_int; ++i) {
		dp[i] = i;
		for (int j = 0; j <= i; ++j) {
			if (s[i] == s[j] && (i - j < 2 || dp1[j + 1][i - 1])) {
				dp1[j][i] = true;
				dp[i] = j == 0 ? 0 : min(dp[i], dp[j] + 1);
			}
		}
	}
	return dp[m_int - 1];
	//动态规划分开
	/*m_int = s.size();
	vector<vector<bool>> dp1(m_int, vector<bool>(m_int, 0));
	for (int i = 0; i < m_int; ++i) {
		for (int j = 0; j <= i; ++j) {
			if (s[i] == s[j] && (i - j < 2 || dp1[j + 1][i - 1]))
				dp1[j][i] = true;
		}
	}
	vector<int> dp(m_int);
	for (int i = 0; i < m_int; ++i) dp[i] = i;
	for (int i = 1; i < m_int; ++i) {
		if (dp1[0][i]) {
			dp[i] = 0;
			continue;
		}
		for (int j = 0; j < i; ++j) {
			if (dp1[j + 1][i])
				dp[i] = min(dp[i], dp[j] + 1);
		}
	}
	return dp[m_int - 1];*/
}

Node * CAlgorithmclass::cloneGraph(Node * node)
{
	//深度搜索
	/*if (node == nullptr)
		return node;
	if (m_NodeMap.find(node) != m_NodeMap.end())
		return m_NodeMap[node];
	Node* newnode = new Node(node->val);
	m_NodeMap[node] = newnode;
	vector<Node*> temp = node->neighbors;
	for (int i = 0; i < temp.size(); ++i)
	{
		newnode->neighbors.push_back(cloneGraph(temp[i]));
	}
	return newnode;*/
	//广度搜索
	if (node == NULL) return node;
	m_NodeMap[node] = new Node(node->val);
	queue<Node*> q;
	q.push(node);
	while (!q.empty()) {
		Node* temp = q.front();
		q.pop();
		for (auto& j : temp->neighbors) {
			if (m_NodeMap.find(j) == m_NodeMap.end()) {
				q.push(j);
				m_NodeMap[j] = new Node(j->val);
			}
			m_NodeMap[temp]->neighbors.push_back(m_NodeMap[j]);
		}
	}
	return m_NodeMap[node];
}

int CAlgorithmclass::canCompleteCircuit(vector<int>& gas, vector<int>& cost)
{
	/*int car(0);
	for (int i = 0; i < gas.size(); ++i)
	{
		if (gas[i] < cost[i]) continue;
		for (int k = i; k < gas.size(); ++k)
		{
			car = car + gas[k] - cost[k];
			if (car < 0) break;
		}
		for (int k = 0; k < i&&car>=0; ++k)
		{
			car = car + gas[k] - cost[k];
			if (car < 0) break;
		}
		if (car >= 0) return i;
		car = 0;
	}
	return -1;*/
	//一次遍历的方法
	int car_tank(0), totle_tank(0), start(0);
	for (int i = 0; i < gas.size(); ++i)
	{
		int dir = gas[i] - cost[i];
		car_tank += dir;
		totle_tank += dir;
		if (car_tank < 0) {
			start = i + 1;
			car_tank = 0;
		}
	}
	return totle_tank < 0 ? -1 : start;
}

int CAlgorithmclass::candy(vector<int>& ratings)
{
	//第一步，先找到当前数组的所有最低峰值
	//第二步，从每个最低峰值往两边遍历，分发糖果，之后总和
	/*if (ratings.size() == 1) return 1;
	int sSize = ratings.size();
	vector<int> minpeak;
	for (int i = 0; i < sSize; ++i) {
		if ((i == 0 && ratings[i] <= ratings[i + 1]) || i == sSize - 1 && ratings[i] <= ratings[i - 1])
			minpeak.push_back(i);
		else if (i>0 && i<sSize - 1 && ratings[i] <= ratings[i - 1] && ratings[i] <= ratings[i + 1])
			minpeak.push_back(i);
	}
	int totle = 0, temp = -1, last = 0;
	for (int i = 0; i < minpeak.size(); ++i) {
		int cur_kid_1 = 0, cur_kid_2 = 0;
		int m = minpeak[i], n = m;
		for (; m == minpeak[i] || (m<sSize&&ratings[m - 1] < ratings[m]); ++m)
			totle += ++cur_kid_1;
		for (; n == minpeak[i] || (n >= 0 && ratings[n + 1] < ratings[n]); --n)
			totle += ++cur_kid_2;
		totle--;
		if (temp == n + 1)
			totle -= cur_kid_2>last ? last : cur_kid_2;
		temp = m - 1;
		last = cur_kid_1;
	}
	return totle;*/
	//优化贪心 
	if (ratings.size() == 1) return 1;
	vector<int> temp(ratings.size(), 1);
	for (int i = 1; i < ratings.size(); ++i)
	{
		if (ratings[i] > ratings[i - 1])
			temp[i] = temp[i - 1] + 1;
	}
	int sum = temp[ratings.size() - 1];
	for (int i = ratings.size() - 2; i >= 0; --i)
	{
		if (ratings[i] > ratings[i + 1])
			temp[i] = max(temp[i], (temp[i + 1] + 1));
		sum += temp[i];
	}
	return sum;
}

int CAlgorithmclass::singleNumber(vector<int>& nums)
{
	/*unordered_map<int, int> amap;
	for (int i = 0; i < nums.size(); ++i)
		amap[nums[i]]++;
	for (unordered_map<int,int>::iterator i = amap.begin(); i != amap.end(); ++i)
		if (i->second == 1) return i->first;
	return 0;*/
	//使用按位异或运算
	int ans(0);
	for (auto& i : nums)
		ans ^= i;
	return ans;
}

int CAlgorithmclass::singleNumber2(vector<int>& nums)
{
	//按位运算  求反，异或，与
	int seenOnce(0), seeTwice(0);
	for (const auto& i : nums) {
		seenOnce = ~seeTwice&(seenOnce^i);
		seeTwice = ~seenOnce&(seeTwice^i);
	}
	return seenOnce;
}

vector<int> CAlgorithmclass::singleNumber3(vector<int>& nums)
{
	//位运算
	int sign = 0, res = 0;
	for (auto& i : nums)
		sign ^= i;
	int temp = sign & (-sign);//n&(n-1)表示保留n的二进制最后一个1
	for (auto&i : nums) {
		if (temp&i)
			res ^= i;
	}
	return{ res,res^sign };
}

Node_random * CAlgorithmclass::copyRandomList(Node_random * head)
{
	/*unordered_map<Node_random*, Node_random*> amap;
	Node_random* newhead = head;
	Node_random* ans = head;
	while (head)
	{
		amap[head] = new Node_random(head->val);
		head = head->next;
	}
	while (newhead)
	{
		amap[newhead]->next = amap[newhead->next];
		amap[newhead]->random = amap[newhead->random];
		newhead = newhead->next;
	}
	return amap[ans];*/
	//空间优化 原地修改
	if (head == nullptr)
		return head;
	//遍历原链表 遍历过程中插入新副本节点
	Node_random* cur = head;
	while (cur) {
		Node_random* node = new Node_random(cur->val);
		Node_random* next = cur->next;
		node->next = next;
		cur->next = node;
		cur = next;
	}
	//遍历原链表 对新副本节点设置random指针
	cur = head;
	while (cur) {
		cur->next->random = cur->random ? cur->random->next : nullptr;
		cur = cur->next->next;
	}
	//分离出原链表与新副本链表
	cur = head;
	Node_random* new_cur = head->next;
	Node_random* res = new_cur;
	while (cur) {
		cur->next = cur->next->next;
		cur = cur->next;

		new_cur->next = cur ? cur->next : nullptr;
		new_cur = new_cur->next;
	}
	return res;
}

bool CAlgorithmclass::wordBreak(string s, vector<string>& wordDict)
{
	//
	/*for (const auto& i : wordDict) m_setstr.insert(i);
	vector<int> sign(1, 0);
	bool ans;
	for (int i = 0; i < s.size(); ++i) {
		int sSize = sign.size();
		for (int j = 0; j<sSize; ++j) {
			if (sign[j]<s.size() && m_setstr.find(s.substr(sign[j], i - sign[j] + 1)) != m_setstr.end()) {
				ans = 1;
				sign.push_back(i + 1);
				break;
			}
			else ans = 0;
		}
	}
	return ans;*/
	//动态规划
	/*for (const auto& i:wordDict)
		m_setstr.insert(i);
	vector<bool> dp(s.size()+1,0);
	dp[0] = 1;
	for (int i = 1; i <= s.size(); ++i)
	{
		for (int j = 0;j<i;++j)
		{
			if (dp[j]&&m_setstr.find(s.substr(j, i - j)) != m_setstr.end()) {
				dp[i] = 1;
				break;
			}
		}
	}
	return dp[s.size()];*/
	//记忆化回溯递归
	vector<bool> dp(s.size(), 0);
	for (const auto& i : wordDict)
		m_setstr.insert(i);
	return back_wordBreak(s, 0, dp);
}

vector<string> CAlgorithmclass::wordBreak2(string s, vector<string>& wordDict)
{
	for (const auto& i : wordDict)
		m_setstr.insert(i);
	vector<bool> dp(s.size() + 1, 0);
	dp[0] = 1;
	for (int i = 1; i <= s.size(); ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			if (dp[j] && m_setstr.find(s.substr(j, i - j)) != m_setstr.end()) {
				dp[i] = 1;
				break;
			}
		}
	}
	if (!dp[s.size()]) return vector<string>();
	string s1 = "";
	back_wordBreak2(s, 0, dp, s1);
	return m_strVt;
}

bool CAlgorithmclass::hasCycle(ListNode * head)
{
	//使用集合set
	/*unordered_set<ListNode*> aset;
	while (head)
	{
		if (aset.find(head) != aset.end())
			return 0;
		aset.insert(head);
		head = head->next;
	}
	return 1;*/
	//奇葩解法
	////随机数 只需要判断这个数是否再次出现
	/*while (head){
		if (head->val == 78)
			return 1;
		else head->val = 78;
		head = head->next;
	}
	return 0;*/
	//快慢指针
	ListNode* slow = head;
	ListNode* fast = head;
	while (fast || fast->next)
	{
		slow = slow->next;
		fast = fast->next->next;
		if (slow == fast) return 1;
	}
	return 0;
}

ListNode * CAlgorithmclass::detectCycle(ListNode * head)
{
	//快慢指针
	/*if (!head || !head->next) return 0;
	ListNode* slow = head;
	ListNode* fast = head;
	while (fast && fast->next)
	{
		slow = slow->next;
		fast = fast->next->next;
		if (slow == fast) {
			ListNode* temp = head;
			while (temp != slow){
				temp = temp->next;
				slow = slow->next;
			}
			return slow;
		}
	}
	return 0;*/
	//判断链表的地址
	if (!head || !head->next)
		return 0;
	while (head) {
		if (head >= head->next)
			return head->next;
		head = head->next;
	}
	return 0;
}

void CAlgorithmclass::reorderList(ListNode * head)
{
	//先用快慢指针找到终中点，将中点之后的链表反转，在插入
	/*if (!head) return;
	ListNode* slow = head;
	ListNode* fast = head;
	while (fast && fast->next)
	{
		slow = slow->next;
		fast = fast->next->next;
	}
	ListNode* end = reverse_list(slow->next);
	slow->next = nullptr;
	while (end)
	{
		ListNode* sec = end->next;
		end->next = head->next;
		head->next = end;
		head = end->next;
		end = sec;
	}*/
	//使用vector容器
	vector<ListNode*> list;
	while (head) {
		list.push_back(head);
		head = head->next;
	}
	int l = 0, r = list.size() - 1;
	while (l < r)
	{
		list[r]->next = list[l]->next;
		list[l++]->next = list[r--];
	}
	list[l]->next = nullptr;
}

ListNode * CAlgorithmclass::insertionSortList(ListNode * head)
{
	/*ListNode* h0 = new ListNode(0);
	ListNode* l = h0;
	ListNode* cur = head;
	while (cur)
	{
		ListNode* temp = cur->next;
		while (l->next&&l->next->val<cur->val)
		{
			l = l->next;
		}
		cur->next = l->next;
		l->next = cur;
		cur = temp;
		l = h0;
	}
	return h0->next;*/
	ListNode* h0 = new ListNode(INT_MIN);
	ListNode* pre = h0;
	ListNode* tail = h0;
	ListNode* cur = head;
	while (cur)
	{
		if (tail->val < cur->val) {
			tail->next = cur;
			tail = cur;
			cur = cur->next;
		}
		else {
			ListNode* temp = cur->next;
			tail->next = temp;
			while (pre->next&&pre->next->val < cur->val)
				pre = pre->next;
			cur->next = pre->next;
			pre->next = cur;
			cur = temp;
			pre = h0;
		}
	}
	return h0->next;
}

ListNode * CAlgorithmclass::sortList(ListNode * head)
{
	//递归的方法
	/*if (!head || !head->next)
		return head;
	ListNode* slow = head;
	ListNode* fast = head->next;
	while (fast&&fast->next){
		slow = slow->next;
		fast = fast->next->next;
	}
	ListNode* temp = slow->next;
	slow->next = nullptr;
	ListNode* l = sortList(head);
	ListNode* r = sortList(temp);
	ListNode* h0 = new ListNode(0);
	ListNode* res = h0;
	while (l&&r){
		if (l->val > r->val) {
			h0->next = r;
			r = r->next;
		}
		else {
			h0->next = l;
			l = l->next;
		}
		h0 = h0->next;
	}
	h0->next = l == nullptr ? r : l;
	return res->next;*/
	//迭代的方法
	if (!head || !head->next) return head;
	int sSize(0), itrv(1);
	ListNode* len = head;
	while (len) {
		sSize++;
		len = len->next;
	}
	ListNode* h0 = new ListNode(0);
	h0->next = head;
	while (itrv < sSize)
	{
		ListNode* pre = h0;
		ListNode* h = h0->next;
		while (h) {
			int i = itrv;
			ListNode* h1 = h;
			for (; h != nullptr && i > 0; i--) {
				h = h->next;
			}
			// i>0说明没有链表2直接返回
			if (i > 0) break;
			ListNode* h2 = h;
			i = itrv;
			for (; h != nullptr && i > 0; i--) {
				h = h->next;
			}
			// 求出两个链表的长度
			int c1 = itrv;
			int c2 = itrv - i;
			//合并
			while (c1 > 0 && c2 > 0) {
				if (h1->val < h2->val) {
					pre->next = h1;
					h1 = h1->next;
					c1--;
				}
				else {
					pre->next = h2;
					h2 = h2->next;
					c2--;
				}
				pre = pre->next;
			}
			pre->next = c1 > 0 ? h1 : h2;
			while (c1 > 0 || c2 > 0) {
				pre = pre->next;
				c1--;
				c2--;
			}
			pre->next = h;
		}
		itrv *= 2;
	}
	return h0->next;
}

int CAlgorithmclass::maxPoints(vector<vector<int>>& points)
{
	//未解决 重复点 的问题
	/*if (points.empty()) return 0;
	int ans = 0;
	long k;
	for (int i = 0; i < points.size(); ++i)
	{
		unordered_map<int, int> amap;
		for (int j = i+1; j < points.size(); ++j)
		{
			if ((points[j][0] - points[i][0]) != 0)
				k = (points[j][1] - points[i][1]) / (points[j][0] - points[i][0]);
			else k = INT_MAX;
			amap[k]++;
		}
		for (unordered_map<int, int>::iterator i = amap.begin(); i != amap.end(); ++i)
		{
			ans = ans > i->second ? ans : i->second;
		}
	}
	return ans+1;*/
	return 0;
}

int CAlgorithmclass::evalRPN(vector<string>& tokens)
{
	stack<int> sta;
	for (string& s : tokens)
	{
		if (41 < s[s.size() - 1] && s[s.size() - 1] < 48) {
			int t1 = sta.top();
			sta.pop();
			int t2 = sta.top();
			sta.pop();
			if (s[0] == 42) sta.push(t1*t2);
			else if (s[0] == 43) sta.push(t1 + t2);
			else if (s[0] == 45) sta.push(t2 - t1);
			else if (s[0] == 47) sta.push(int(t2 / t1));
		}
		else sta.push(atoi(s.c_str()));
	}
	return sta.top();
}

string CAlgorithmclass::reverseWords(string s)
{
	/*if (s.empty()) return s;
	string ans = "";
	for (int i = 0; i < s.size(); ++i)
	{
		if (s[i] == ' ') continue;
		int start = i;
		while (i < s.size())
		{
			if (s[i] == ' ')
				break;
			i++;
		}
		ans = " " + s.substr(start, i - start) + ans;
	}
	if (ans.empty()) return "";
	return ans.substr(1,ans.size()-1);*/
	//原地修改
	reverse(s.begin(), s.end());                        //整体反转
	int start = 0, end = s.size() - 1;
	while (start < s.size() && s[start] == ' ') start++;//首空格
	while (end >= 0 && s[end] == ' ') end--;            //尾空格
	if (start > end) return "";                         //特殊情况
	for (int r = start; r <= end;) {                    //逐单词反转
		while (s[r] == ' '&& r <= end) r++;
		int l = r;
		while (s[l] != ' '&&l <= end) l++;
		reverse(s.begin() + r, s.begin() + l);
		r = l;
	}
	int tail = start;                                   //处理中间冗余空格
	for (int i = start; i <= end; i++) {
		if (s[i] == ' '&&s[i - 1] == ' ') continue;
		s[tail++] = s[i];
	}
	return s.substr(start, tail - start);
}

int CAlgorithmclass::maxProduct(vector<int>& nums)
{
	int ans(INT_MIN), imin(1), imax(1);
	for (int i = 0; i < nums.size(); ++i) {
		if (nums[i] < 0)
			swap(imax, imin);
		imax = max(nums[i], nums[i] * imax);
		imin = min(nums[i], nums[i] * imin);
		ans = max(ans, imax);
	}
	return ans;
}

int CAlgorithmclass::findMin(vector<int>& nums)
{
	//自己的二分法
	/*if (nums.size() == 1) return nums[0];
	int l = 0, r = nums.size() - 1;
	while (l<r)
	{
		int mid = (l + r) / 2;
		if (nums[mid] > nums[mid + 1])
			return nums[mid+1];
		if (nums[mid] > nums[r])
			l = mid;
		else if (nums[mid] < nums[l])
			r = mid;
		else {
			l++;
			r--;
		}
	}
	return nums[0];*/
	//别人的二分法
	int l = 0, r = nums.size() - 1;
	while (l < r)
	{
		int mid = l + (r - l) / 2;
		if (nums[mid] > nums[r])
			l = mid + 1;
		else
			r = mid;
	}
	return nums[l];
}

int CAlgorithmclass::findMin2(vector<int>& nums)
{
	int l = 0, r = nums.size() - 1;
	while (l < r)
	{
		/*while (r>0 && nums[r] == nums[r - 1])
			r--;
		while (l<nums.size()-1 && nums[l] == nums[l + 1])
			l++;*/
		int mid = l + (r - l) / 2;
		if (nums[mid] > nums[r])
			l = mid + 1;
		else if (nums[mid] < nums[r])
			r = mid;
		else
			r = mid - 1;
	}
	return nums[l];
}

ListNode * CAlgorithmclass::getIntersectionNode(ListNode * headA, ListNode * headB)
{
	/*unordered_set<ListNode*> aset;
	while (headB)
	{
		aset.insert(headB);
		headB = headB->next;
	}
	while (headA)
	{
		if (aset.find(headA) != aset.end())
			return headA;
		headA = headA->next;
	}*/
	ListNode* h1 = headA;
	ListNode* h2 = headB;
	while (h1 != h2)
	{
		h1 = h1 != nullptr ? h1->next : h2;
		h2 = h2 != nullptr ? h2->next : h1;
	}
	return h1;
}

int CAlgorithmclass::findPeakElement(vector<int>& nums)
{
	int l = 0, r = nums.size() - 1;
	while (l < r)
	{
		int mid = l + (r - l) / 2;
		if (nums[mid] > nums[mid + 1]) r = mid;
		else l = mid + 1;
	}
	return l;
}

int CAlgorithmclass::compareVersion(string version1, string version2)
{
	int i = 0, j = 0, v1, v2;
	while (i < version1.size() && j < version2.size())
	{
		for (; i < version1.size(); ++i)
		{
			int start = i;
			while (i < version1.size())
			{
				if (version1[i] == '.') break;
				i++;
			}
			v1 = atoi(version1.substr(start, i - start).c_str());
			break;
		}
		for (; j < version2.size(); ++j)
		{
			int start = j;
			while (j < version2.size())
			{
				if (version2[j] == '.') break;
				j++;
			}
			v2 = atoi(version2.substr(start, j - start).c_str());
			break;
		}
		if (v1 != v2) return v1 > v2 ? 1 : -1;
		v1 = 0;
		v2 = 0;
		i++;
		j++;
	}
	return 0;
}

string CAlgorithmclass::fractionToDecimal(int numerator, int denominator)
{
	if (!denominator) return "";
	if (!numerator) return "0";
	string s = "";
	long long num = static_cast<long long>(numerator);
	long long denom = static_cast<long long>(denominator);
	if ((num > 0) ^ (denom > 0))s.push_back('-');
	num = abs(num); denom = abs(denom);
	s.append(to_string(num / denom));
	num %= denom;                         //获得余数
	if (num == 0)return s;             //余数为0，表示整除了，直接返回结果
	s.push_back('.');              //余数不为0，添加小数点
	int index = s.size() - 1;          //获得小数点的下标
	unordered_map<int, int> record;      //map用来记录出现重复数的下标，然后将'('插入到重复数前面就好了
	while (num&&record.count(num) == 0) {   //小数部分：余数不为0且余数还没有出现重复数字
		record[num] = ++index;
		num *= 10;                        //余数扩大10倍，然后求商，和草稿本上运算方法是一样的
		s += to_string(num / denom);
		num %= denom;
	}
	if (record.count(num) == 1) {           //出现循环余数，我们直接在重复数字前面添加'(',字符串末尾添加')'
		s.insert(record[num], "(");
		s.push_back(')');
	}
	return s;
}

vector<int> CAlgorithmclass::twoSum(vector<int>& numbers, int target)
{
	int l = 0, r = numbers.size() - 1;
	while (l < r)
	{
		int mid = l + (r - l) / 2;
		if (numbers[mid] > target)
			r = mid;
		else
			l = mid + 1;
	}
	r = l;
	l = 0;
	while (l < r)
	{
		int sum = numbers[l] + numbers[r];
		if (sum == target) return{ l + 1,r + 1 };
		else if (sum > target) r--;
		else l++;
	}
	/*unordered_map<int, int> amap;
	for (int i = 0; i <= l; i++)
	{
		if (amap.find(target - numbers[i]) != amap.end())
			return{ amap[target - numbers[i]],i + 1 };
		amap[numbers[i]] = i + 1;
	}*/
	return vector<int>();
}

string CAlgorithmclass::convertToTitle(int n)
{
	string s = "";
	while (n > 0)
	{
		n--;
		s = char(65 + (n % 26)) + s;
		n /= 26;
	}
	return s;
}

int CAlgorithmclass::majorityElement(vector<int>& nums)
{
	//哈希表
	/*unordered_map<int, int> amap;
	int sSize = nums.size()/2;
	for (int i = 0; i < nums.size(); ++i)
	{
		if (++amap[nums[i]] > sSize)
			return nums[i];
	}*/
	//排序法
	/*sort(nums.begin(), nums.end());
	return nums[nums.size()/2];*/
	//投票算法
	int candidate = -1, count = 0;
	for (auto& i : nums) {
		if (i == candidate)
			count++;
		else if (--count < 0) {
			candidate = i;
			count = 1;
		}
	}
	return candidate;
}

int CAlgorithmclass::titleToNumber(string s)
{
	int ans = 0, temp = 1;
	for (int i = s.size() - 1; i >= 0; --i)
	{
		ans += int(s[i] - 64)*temp;
		if (i != 0) temp *= 26;
	}
	return ans;
}

int CAlgorithmclass::trailingZeroes(int n)
{
	int five = 0;
	while (n > 5)
	{
		five += n / 5;
		n /= 5;
	}
	return five;
}

int CAlgorithmclass::calculateMinimumHP(vector<vector<int>>& dungeon)
{
	int row = dungeon.size(), col = dungeon[0].size();
	dungeon[row - 1][col - 1] = -dungeon[row - 1][col - 1] < 0 ? 0 : -dungeon[row - 1][col - 1];
	for (int i = col - 2; i >= 0; --i)
	{
		dungeon[row - 1][i] = max(0, dungeon[row - 1][i + 1]) - dungeon[row - 1][i];
	}
	for (int i = row - 2; i >= 0; --i)
	{
		dungeon[i][col - 1] = max(0, dungeon[i + 1][col - 1]) - dungeon[i][col - 1];
	}
	for (int i = row - 2; i >= 0; --i)
	{
		for (int j = col - 2; j >= 0; --j)
		{
			dungeon[i][j] = max(0, min(dungeon[i][j + 1], dungeon[i + 1][j])) - dungeon[i][j];
		}
	}
	return 1 + dungeon[0][0];
}

string CAlgorithmclass::largestNumber(vector<int>& nums)
{
	vector<string> svt(nums.size());
	string ans = "";
	for (int i = 1; i < nums.size(); ++i)
		svt[i] = to_string(nums[i]);
	sort(svt.begin(), svt.end(), [](string& s1, string& s2) { return s1 + s2 > s2 + s1; });
	for (int i = 0; i < svt.size(); ++i)
		ans += svt[i];
	return ans;
}

vector<string> CAlgorithmclass::findRepeatedDnaSequences(string s)
{
	/*unordered_map<string, int> amap;
	vector<string> ans;
	for (int i = 9; i < s.size(); ++i) amap[s.substr(i - 9, 10)]++;
	for (unordered_map<string, int>::iterator i = amap.begin(); i != amap.end(); ++i) if (i->second > 1) ans.push_back(i->first);
	return ans;*/
	//字母转数字 并用位运算滑动窗口
	unordered_map<char, int> m{ { 'A', 0 },{ 'C', 1 },{ 'G', 2 },{ 'T', 3 } };
	vector<string> res;
	bitset<1 << 20> s1, s2; //那么所有组合的值将在0到(1 << 20 - 1)之间
	int val = 0, mask = (1 << 20) - 1; //mask等于二进制的20个1
									   //类似与滑动窗口先把前10个字母组合
	for (int i = 0; i < 10; ++i) val = (val << 2) | m[s[i]];
	s1.set(val); //置位
	for (int i = 10; i < s.size(); ++i) {
		val = ((val << 2) & mask) | m[s[i]]; //去掉左移的一个字符再加上一个新字符
		if (s2.test(val)) continue; //出现过两次跳过
		if (s1.test(val)) {
			res.push_back(s.substr(i - 9, 10));
			s2.set(val);
		}
		else s1.set(val);
	}
	return res;
}

void CAlgorithmclass::rotate(vector<int>& nums, int k)
{
	/*int sSize = nums.size();
	if (k == sSize || sSize == 1) return;
	if (k > sSize) k -= sSize;
	if (k <= sSize / 2) {
		while (k--)
		{
			for (int i = sSize - 2; i >= 0; --i)
			{
				swap(nums[i], nums[i + 1]);
			}
		}
	}
	else {
		while (k--)
		{
			for (int i = 1;i<sSize;++i)
			{
				swap(nums[i], nums[i - 1]);
			}
		}
	}*/
	//三次反转
	/*int sSize = nums.size();
	if (k == sSize || sSize == 1) return;
	k = k%sSize;
	reverse(nums.begin(),nums.begin()+sSize-k);
	reverse(nums.begin() + sSize - k, nums.end());
	reverse(nums.begin(),nums.end());*/
	//
	int sSize = nums.size();
	if (k == sSize || sSize == 1) return;
	k = k % sSize;
	int count = 0;
	for (int start = 0; count < nums.size(); ++start) {
		int current = start;
		int prev = nums[start];
		do {
			int next = (current + k) % nums.size();
			int temp = nums[next];
			nums[next] = prev;
			prev = temp;
			current = next;
			count++;
		} while (start != current);
	}
}

uint32_t CAlgorithmclass::reverseBits(uint32_t n)
{
	//逐个反转
	/*uint32_t temp = 31,ans;
	while (n != 0)
	{
		ans += (n & 1) << temp;
		n = n >> 1;
		temp--;
	}
	return ans;*/
	//带记忆化的按字节颠倒
	/*uint32_t ans = 0, temp = 24;
	unordered_map<uint32_t, uint32_t> amap;
	while (n!= 0)
	{
		ans += reverseByte(n & 0xff,amap) << temp;
		n = n >> 8;
		temp -= 8;
	}
	return ans;*/
	//只是用位运算  //  0xff00ff00
	n = (n >> 16) | (n << 16);
	n = ((n & 0xff00ff00) >> 8) | ((n & 0x00ff00ff) << 8);
	n = ((n & 0xf0f0f0f0) >> 4) | ((n & 0x0f0f0f0f) << 4);
	n = ((n & 0xcccccccc) >> 2) | ((n & 0x33333333) << 2);
	n = ((n & 0xaaaaaaaa) >> 1) | ((n & 0x55555555) << 1);
	return n;
}

int CAlgorithmclass::hammingWeight(uint32_t n)
{
	/*int ans = 0;
	while (n!= 0)
	{
		ans += (n & 1);
		n = n >> 1;
	}
	return ans;*/
	//小技巧
	/*int ans = 0;
	while (n!=0)
	{
		ans++;
		n &= n - 1;
	}
	return ans;*/
	//记忆化  优化
	unordered_map<uint32_t, int> amap;
	int ans = 0;
	while (n != 0)
	{
		ans += hamming_Weight(n & 0xff, amap);
		n = n >> 8;
	}
	return ans;
}

vector<double> CAlgorithmclass::intersection(vector<int>& start1, vector<int>& end1, vector<int>& start2, vector<int>& end2)
{
	vector<double> ans;
	int x1 = start1[0], y1 = start1[1];
	int x2 = end1[0], y2 = end1[1];
	int x3 = start2[0], y3 = start2[1];
	int x4 = end2[0], y4 = end2[1];
	if ((y2 - y1)*(x3 - x4) == (x2 - x1)*(y3 - y4)) {
		if ((y2 - y1)*(x3 - x1) == (x2 - x1)*(y3 - y1)) {
			if (intersection_inside(x1, y1, x2, y2, x3, y3)) {
				intersection_update(ans, (double)x3, (double)y3);
			}
			if (intersection_inside(x1, y1, x2, y2, x4, y4)) {
				intersection_update(ans, (double)x4, (double)y4);
			}
			if (intersection_inside(x3, y3, x4, y4, x1, y1)) {
				intersection_update(ans, (double)x1, (double)y1);
			}
			if (intersection_inside(x3, y3, x4, y4, x2, y2)) {
				intersection_update(ans, (double)x2, (double)y2);
			}
		}
	}
	else {
		double t1 = (double)(x3 * (y4 - y3) + y1 * (x4 - x3) - y3 * (x4 - x3) - x1 * (y4 - y3)) / ((x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1));
		double t2 = (double)(x1 * (y2 - y1) + y3 * (x2 - x1) - y1 * (x2 - x1) - x3 * (y2 - y1)) / ((x4 - x3) * (y2 - y1) - (x2 - x1) * (y4 - y3));
		// 判断 t1 和 t2 是否均在 [0, 1] 之间
		if (t1 >= 0.0 && t1 <= 1.0 && t2 >= 0.0 && t2 <= 1.0) {
			ans = { x1 + t1 * (x2 - x1), y1 + t1 * (y2 - y1) };
		}
	}
	return ans;
}

int CAlgorithmclass::rob(vector<int>& nums)
{
	/*if (nums.size() == 0) return 0;
	if (nums.size() == 1) return nums[0];
	vector<int> dp(nums.size());
	int ans = 0,temp = 0;
	dp[0] = nums[0];
	dp[1] = nums[1];
	ans = max(nums[0], nums[1]);
	temp = nums[0];
	for (int i = 2; i < nums.size(); ++i)
	{
		dp[i] = max(dp[i - 2],temp)+nums[i];
		temp = max(dp[i - 1], temp);
		ans = max(dp[i], ans);
	}
	return ans;*/
	//优化后
	int pre = 0, cur = 0;
	for (int i = 0; i < nums.size(); ++i)
	{
		int temp = cur;
		cur = max(cur, nums[i] + pre);
		pre = temp;
	}
	return cur;
}

vector<int> CAlgorithmclass::rightSideView(TreeNode * root)
{
	//广度优先搜索
	/*if (!root) return{};
	vector<int> ans;
	queue<TreeNode*> q;
	q.push(root);
	while (!q.empty())
	{
		int sSize = q.size();
		for (int i = 0; i < sSize; ++i)
		{
			TreeNode* temp = q.front();
			q.pop();
			if (temp->left) q.push(temp->left);
			if (temp->right) q.push(temp->right);
			if (i == sSize - 1) ans.push_back(temp->val);
		}
	}
	return ans;*/
	//深度优先搜索
	back_rightSideView(root, 0);
	return m_intVt;
}

int CAlgorithmclass::numIslands(vector<vector<char>>& grid)
{
	//深度搜索
	/*int ans = 0;
	m_pairVt = { {1,0},{-1,0},{0,1},{0,-1} };
	m_row = grid.size();
	m_col = grid[0].size();
	for (int i = 0; i < m_row; ++i)
	{
		for (int j = 0; j < m_col; ++j)
		{
			if (grid[i][j] == '1') {
				back_numIslands(grid, i, j);
				ans++;
			}
		}
	}
	return ans;*/
	//广度搜索
	int nr = grid.size();
	if (!nr) return 0;
	int nc = grid[0].size();

	int num_islands = 0;
	for (int r = 0; r < nr; ++r) {
		for (int c = 0; c < nc; ++c) {
			if (grid[r][c] == '1') {
				++num_islands;
				grid[r][c] = '0'; // mark as visited
				queue<pair<int, int>> ors;
				ors.push({ r, c });
				while (!ors.empty()) {
					auto rc = ors.front();
					ors.pop();
					int row = rc.first, col = rc.second;
					if (row - 1 >= 0 && grid[row - 1][col] == '1') {
						ors.push({ row - 1, col }); grid[row - 1][col] = '0';
					}
					if (row + 1 < nr && grid[row + 1][col] == '1') {
						ors.push({ row + 1, col }); grid[row + 1][col] = '0';
					}
					if (col - 1 >= 0 && grid[row][col - 1] == '1') {
						ors.push({ row, col - 1 }); grid[row][col - 1] = '0';
					}
					if (col + 1 < nc && grid[row][col + 1] == '1') {
						ors.push({ row, col + 1 }); grid[row][col + 1] = '0';
					}
				}
			}
		}
	}

	return num_islands;
}

int CAlgorithmclass::rangeBitwiseAnd(int m, int n)
{
	//一次遍历 超时
	/*int res = m+1;
	while (res <= n)
		m &= res++;
	return m;*/
	//位运算
	/*int res = 0;
	while (m != n) {
		m >>= 1;
		n >>= 1;
		res++;
	}
	return m << res;*/
	//一种方法：把二进制最右边的1置位0
	while (m < n)
		n &= n - 1;
	return n;
}

bool CAlgorithmclass::isHappy(int n)
{
	//哈希表 方法
	/*unordered_set<int> aset;
	aset.insert(n);
	while (true)
	{
		int res = 0;
		while (n)
		{
			int yu = n % 10;
			res += yu*yu;
			n /= 10;
		}
		if (res == 1||res == 7) return true;
		if (aset.find(res) != aset.end()) return false;
		aset.insert(res);
		n = res;
	}
	return false;*/
	//不用哈希
	/*while (n>=10)
	{
		int res = 0;
		while (n)
		{
			int yu = n % 10;
			res += yu*yu;
			n /= 10;
		}
		n = res;
	}
	return n == 1 || n == 7;*/
	//快慢指针
	int slow = n;
	int fast = getnext(n);
	while (fast != 1 && fast != slow)
	{
		slow = getnext(slow);
		fast = getnext(getnext(fast));
	}
	return fast == 1;
}

ListNode * CAlgorithmclass::removeElements(ListNode * head, int val)
{
	ListNode* temp = new ListNode(0);
	temp->next = head;
	ListNode* ans = temp;
	ListNode* todelete = nullptr;
	while (head)
	{
		if (head->val == val) {
			temp->next = head->next;
			todelete = head;
		}
		else temp = head;
		head = head->next;
		if (todelete) {
			delete todelete;
			todelete = nullptr;
		}
	}
	ListNode* res = ans->next;
	delete ans;
	return res;
}

int CAlgorithmclass::countPrimes(int n)
{

	vector<bool> isPrime(n + 1, 1);
	int count = 0;
	for (int i = 2; i < n; ++i) {
		if (isPrime[i]) {
			++count;
			for (int j = 2; i*j < n; ++j)
				isPrime[i*j] = 0;
		}
	}
	return count;
}

ListNode * CAlgorithmclass::addTwoNumbers2(ListNode * l1, ListNode * l2)
{
	//反转链表
	/*ListNode* _l1 = reverse_list(l1);
	ListNode* _l2 = reverse_list(l2);
	l1 = _l1;
	l2 = _l2;
	ListNode* res = l1;
	int temp = 0;
	while (_l1&&_l2) {
		temp += (_l1->val + _l2->val);
		int yu = temp % 10;
		temp /= 10;
		_l1->val = yu;
		_l1 = _l1->next;
		_l2 = _l2->next;
	}
	if (!_l1) {
		while (res->next) res = res->next;
		res->next = _l2;
		_l1 = res->next;
	};
	while (_l1) {
		temp += _l1->val;
		int yu = temp % 10;
		temp /= 10;
		_l1->val = yu;
		_l1 = _l1->next;
	}
	if (temp == 0) return reverse_list(l1);
	else {
		while (res->next) res = res->next;
		res->next = new ListNode(temp);
	}
	return reverse_list(l1);*/
	//使用栈
	stack<int> s1, s2;
	while (l1) {
		s1.push(l1->val);
		l1 = l1->next;
	}
	while (l2) {
		s2.push(l2->val);
		l2 = l2->next;
	}
	int carry = 0;
	ListNode* ans = nullptr;
	while (!s1.empty() || !s2.empty() || carry != 0) {
		int a = s1.empty() ? 0 : s1.top();
		int b = s2.empty() ? 0 : s2.top();
		if (!s1.empty()) s1.pop();
		if (!s2.empty()) s2.pop();
		int cur = a + b + carry;
		carry = cur / 10;
		cur %= 10;
		auto curnode = new ListNode(cur);
		curnode->next = ans;
		ans = curnode;
	}
	return ans;
}

bool CAlgorithmclass::isIsomorphic(string s, string t)
{
	//使用哈希表
	/*unordered_map<char, char> amap, tmap;
	for (int i = 0; i < s.size(); ++i) {
		if (amap.find(s[i]) != amap.end()) {
			if (amap[s[i]] == t[i]) continue;
			else return 0;
		}
		amap[s[i]] = t[i];
	}
	for (int i = 0; i < t.size(); ++i) {
		if (tmap.find(t[i]) != tmap.end()) {
			if (tmap[t[i]] == s[i]) continue;
			else return 0;
		}
		tmap[t[i]] = s[i];
	}
	return 1;*/
	//不使用哈希表
	vector<int> sVt(256, 0), tVt(256, 0);
	for (int i = 0; i < s.size(); ++i)
	{
		/*if (sVt[int(s[i])]!= 0 && sVt[int(s[i])] != t[i])return 0;
		if (tVt[int(t[i])] != 0 && tVt[int(t[i])] != s[i])return 0;
		sVt[int(s[i])] = int(t[i]);
		tVt[int(t[i])] = int(s[i]);*/
		if (sVt[int(s[i])] != tVt[int(t[i])]) return 0;
		sVt[int(s[i])] = i + 1;
		tVt[int(t[i])] = i + 1;
	}
	return 1;
}

bool CAlgorithmclass::canFinish(int numCourses, vector<vector<int>>& prerequisites)
{
	queue<int> q;
	vector<int> indegrees(numCourses, 0);
	vector<vector<int>> adjacency(numCourses);
	for (int i = 0; i < prerequisites.size(); ++i)
	{
		indegrees[prerequisites[i][0]]++;
		adjacency[prerequisites[i][1]].push_back(prerequisites[i][0]);
	}
	for (int i = 0; i < numCourses; ++i)
	{
		if (indegrees[i] == 0)
			q.push(i);
	}
	while (!q.empty())
	{
		int temp = q.front();
		q.pop();
		numCourses--;
		for (const auto& j : adjacency[temp])
		{
			if (--indegrees[j] == 0)
				q.push(j);
		}
	}
	return numCourses == 0;
}

vector<vector<int>> CAlgorithmclass::updateMatrix(vector<vector<int>>& matrix)
{
	m_pairVt = { {1,0},{-1,0},{0,1},{0,-1} };
	m_row = matrix.size();
	m_col = matrix[0].size();
	//深度搜索 递归
	/*for (int i = 0; i < m_row; ++i)
	{
		for (int j = 0; j < m_col; ++j)
		{
			if (matrix[i][j] == 1 && !((i > 0 && matrix[i - 1][j] == 0)
				|| (i < m_row - 1 && matrix[i + 1][j] == 0)
				|| (j > 0 && matrix[i][j - 1] == 0)
				|| (j < m_col - 1 && matrix[i][j + 1] == 0)))
				matrix[i][j] = m_row + m_col;
		}
	}
	for (int i = 0; i < m_row; ++i)
	{
		for (int j = 0; j < m_col; ++j)
		{
			if (matrix[i][j] == 1)
				back_updateMatrix(matrix,i,j);
		}
	}
	return matrix;*/
	//广度搜索
	/*queue<pair<int, int>> q;
	vector<vector<int>> seen(m_row, vector<int>(m_col));
	for (int i = 0; i < m_row; ++i)
	{
		for (int j = 0; j < m_col; ++j)
		{
			if (matrix[i][j] == 0) {
				q.push(make_pair(i, j));
				seen[i][j] = 1;
			}
		}
	}
	while (!q.empty())
	{
		pair<int, int> cur = q.front(); q.pop();
		for (int k = 0; k < 4; ++k)
		{
			int temp_i = cur.first + m_pairVt[k].first;
			int temp_j = cur.second + m_pairVt[k].second;
			if (temp_i < 0 || temp_j<0 || temp_i >= m_row || temp_j >= m_col || seen[temp_i][temp_j])
				continue;
			matrix[temp_i][temp_j] = matrix[cur.first][cur.second] + 1;
			q.push(make_pair(temp_i,temp_j));
			seen[temp_i][temp_j] = 1;
		}
	}
	return matrix;*/

	//动态规划
	vector<vector<int>> dist(m_row, vector<int>(m_col, INT_MAX / 2));
	for (int i = 0; i < m_row; ++i) {
		for (int j = 0; j < m_col; ++j) {
			if (matrix[i][j] == 0) {
				dist[i][j] = 0;
			}
		}
	}
	for (int i = 0; i < m_row; ++i) {
		for (int j = 0; j < m_col; ++j) {
			if (i - 1 >= 0) dist[i][j] = min(dist[i][j], dist[i - 1][j] + 1);
			if (j - 1 >= 0) dist[i][j] = min(dist[i][j], dist[i][j - 1] + 1);
		}
	}
	for (int i = m_row - 1; i >= 0; --i) {
		for (int j = m_col - 1; j >= 0; --j) {
			if (i + 1 < m_row) dist[i][j] = min(dist[i][j], dist[i + 1][j] + 1);
			if (j + 1 < m_col) dist[i][j] = min(dist[i][j], dist[i][j + 1] + 1);
		}
	}
	return dist;
}

int CAlgorithmclass::minSubArrayLen(int s, vector<int>& nums)
{
	//O(n)的解法
	/*int l = 0, r = 0,ans = INT_MAX,temp = 0;
	while(r< nums.size())
	{
		temp += nums[r];
		while (l <= r&&temp >= s)
		{
			ans = min(ans, r - l + 1);
			temp -= nums[l++];
		}
		r++;
	}
	return ans ==INT_MAX?0:ans;*/
	//O(n*logn)的解法
	vector<int> sum(nums.size() + 1, 0);
	int ans = INT_MAX;
	for (int i = 1; i <= nums.size(); ++i)
		sum[i] = sum[i - 1] + nums[i - 1];
	for (int i = 1; i <= nums.size(); ++i)
	{
		int temp = sum[i - 1] + s;
		auto bound = lower_bound(sum.begin(), sum.end(), temp);
		if (bound != sum.end())
			ans = min(ans, static_cast<int>(bound - (sum.begin() + i - 1)));
	}
	return ans == INT_MAX ? 0 : ans;
}

vector<vector<int>> CAlgorithmclass::merge(vector<vector<int>>& intervals)
{
	if (intervals.size() < 2) return intervals;
	sort(intervals.begin(), intervals.end());
	vector<vector<int>> ans;
	vector<int> res = intervals[0];
	for (int i = 1; i < intervals.size(); ++i)
	{
		if (intervals[i][0] > res[1]) {
			ans.push_back(res);
			res = intervals[i];
			continue;
		}
		else if (intervals[i][1] > res[1]) {
			res[1] = intervals[i][1];
		}
	}
	ans.push_back(res);
	return ans;
}

vector<int> CAlgorithmclass::findOrder(int numCourses, vector<vector<int>>& prerequisites)
{
	vector<int> ans;
	queue<int> q;
	vector<int> indegrees(numCourses, 0);
	vector<vector<int>> adjacency(numCourses);
	for (int i = 0; i < prerequisites.size(); ++i)
	{
		indegrees[prerequisites[i][0]]++;
		adjacency[prerequisites[i][1]].push_back(prerequisites[i][0]);
	}
	for (int i = 0; i < numCourses; ++i)
	{
		if (indegrees[i] == 0)
			q.push(i);
	}
	while (!q.empty())
	{
		int temp = q.front();
		q.pop();
		numCourses--;
		ans.push_back(temp);
		for (const auto& j : adjacency[temp])
		{
			if (--indegrees[j] == 0) {
				q.push(j);
			}

		}
	}
	return numCourses == 0 ? ans : vector<int>();
}

bool CAlgorithmclass::canJump(vector<int>& nums)
{
	//递归超出时间限制
	/*return back_canJump(0,nums);*/
	//贪心
	int temp = 0;
	for (int i = 0; i < nums.size(); ++i)
	{
		if (i > temp) return 0;
		temp = max(nums[i] + i, temp);
	}
	return 1;
}

int CAlgorithmclass::rob2(vector<int>& nums)
{
	if (nums.size() == 1) return nums[0];
	int ans = 0, sSize = nums.size(), pre = 0, cur = 0;
	for (int i = 0; i < sSize - 1; ++i)
	{
		int temp = cur;
		cur = max(cur, nums[i] + pre);
		pre = temp;
	}
	ans = cur;
	pre = 0;
	cur = 0;
	for (int i = nums.size() - 1; i > 0; --i)
	{
		int temp = cur;
		cur = max(cur, nums[i] + pre);
		pre = temp;
	}
	return max(ans, cur);
}

int CAlgorithmclass::rob3(TreeNode * root)
{
	vector<int> ans = back_rob3(root);
	return max(ans[0], ans[1]);
}

int CAlgorithmclass::maxArea(vector<int>& height)
{
	int l = 0, r = height.size() - 1, ans = 0;
	while (l < r) {
		if (height[l] < height[r]) ans = max(ans, (r - l)*height[l++]);
		else  ans = max(ans, (r - l)*height[r--]);
	}
	return ans;
}

string CAlgorithmclass::shortestPalindrome(string s)
{
	//中心扩展法
	/*if (!s.size()) return "";
	string ans = s;
	int sign = 0;
	for (int i = s.size() /2-1; i > 0; --i)
	{
		int l = i;
		int r = sign == 1 ? i + 1 : i + 2;
		while (l>=0&&r<s.size())
		{
			if (s[l] != s[r])
				break;
			l--;
			r++;
		}
		if (l < 0) {
			string temp = s.substr(r);
			reverse(temp.begin(), temp.end());
			return temp+ans;
		}
		if (!sign) {
			sign = 1;
			i++;
		}
		else sign = 0;
	}
	reverse(s.begin(), s.end());
	return s+ans.substr(1);*/
	//
	string restr = s;
	int sSize = s.size();
	reverse(restr.begin(), restr.end());
	s = ' ' + s + '#' + restr;//让下标从1开始
	int n = s.size() - 1;//实际长度
	vector<int> ne(n + 1);//next数组
	for (int i = 2, j = 0; i <= n; i++) {//求next数组 
		while (j&&s[i] != s[j + 1]) j = ne[j];
		if (s[i] == s[j + 1]) j++;
		ne[i] = j;
	}
	return s.substr(sSize + 2, sSize - ne[n]) + s.substr(1, sSize);
}

int CAlgorithmclass::findKthLargest(vector<int>& nums, int k)
{
	priority_queue<int, vector<int>, greater<int> > p;
	for (int i = 0; i < nums.size(); ++i)
	{
		if (p.size() >= k && p.top() > nums[i])
			continue;
		p.push(nums[i]);
		if (p.size() > k)
			p.pop();
	}
	return p.top();
}

int CAlgorithmclass::getMaxRepetitions(string s1, int n1, string s2, int n2)
{
	//暴力超时
	/*int ans = 0;
	int i = 0, j = 0;
	int temp = n1;
	int res = n2;
	while (n1)
	{
		if (s1[i] == s2[j]) {
			i++;
			j++;
		}
		else {
			i++;
		}
		if (j >= s2.size()) {
			j = 0;
			n2--;
			if (n2 == 0) {
				ans++;
				n2 = res;
			}
		}
		if (i >= s1.size()) {
			i = 0;
			n1--;
		}
	}
	return ans;*/
	//动态规划
	if (n1 == 0) return 0;
	int s1cnt = 0, s2cnt = 0, index = 0;
	unordered_map<int, pair<int, int>> recall;
	pair<int, int> pre_loop, in_loop;
	while (true)
	{
		++s1cnt;
		for (int i = 0; i < s1.size(); ++i)
		{
			if (s1[i] == s2[index]) {
				index++;
				if (index == s2.size()) {
					++s2cnt;
					index = 0;
				}
			}
		}
		if (s1cnt == n1)
			return s2cnt / n2;
		if (recall.find(index) != recall.end()) {
			pre_loop = recall[index];
			in_loop = make_pair(s1cnt - pre_loop.first, s2cnt - pre_loop.second);
			break;
		}
		else
			recall[index] = make_pair(s1cnt, s2cnt);
	}
	int ans = pre_loop.second + (n1 - pre_loop.first) / in_loop.first * in_loop.second;
	int rest = (n1 - pre_loop.first) % in_loop.first;
	for (int i = 0; i < rest; ++i) {
		for (char ch : s1) {
			if (ch == s2[index]) {
				++index;
				if (index == s2.size()) {
					++ans;
					index = 0;
				}
			}
		}
	}
	return ans / n2;
}

vector<vector<int>> CAlgorithmclass::combinationSum3(int k, int n)
{
	vector<vector<int>> ans;
	vector<int> res(k);
	back_combinationSum3(ans, res, k, n, 0);
	return ans;
}

bool CAlgorithmclass::containsDuplicate(vector<int>& nums)
{
	//哈希表
	/*unordered_set<int> aset;
	for (auto& i : nums) {
		if (aset.find(i) != aset.end())
			return 1;
		aset.insert(i);
	}*/
	//数组
	vector<bool> temp(nums.size() + 1, 0);
	for (int i = 0; i < nums.size(); ++i)
	{
		if (temp[nums[i]])
			return 1;
		temp[nums[i]] = true;
	}
	return false;
}

bool CAlgorithmclass::containsNearbyDuplicate(vector<int>& nums, int k)
{
	//哈希表
	/*unordered_map<int, int> amap;
	for (int i = 0; i < nums.size(); ++i)
	{
		if (amap.find(nums[i]) != amap.end()) {
			if (abs(i - amap[nums[i]]) <= k)
				return 1;
		}
		amap[nums[i]] = i;
	}*/
	//优化哈希表  维护k区间
	unordered_set<int> aset;
	for (int i = 0; i < nums.size(); ++i) {
		if (aset.find(nums[i]) != aset.end()) return 1;
		aset.insert(nums[i]);
		if (aset.size() > k) aset.erase(nums[i - k]);
	}
	return false;
}

bool CAlgorithmclass::containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t)
{
	set<int> aset;
	for (int i = 0; i < nums.size(); ++i) {
		auto x = aset.lower_bound(nums[i]);
		if (x != aset.end() && *x <= nums[i] + t) return 1;
		auto y = aset.upper_bound(nums[i]);
		if (y != aset.begin() && nums[i] <= *--y + t) return 1;
		aset.insert(nums[i]);
		if (aset.size() > k) aset.erase(nums[i - k]);
	}
	return false;
}

vector<vector<int>> CAlgorithmclass::getSkyline(vector<vector<int>>& buildings)
{
	return vector<vector<int>>();
}

int CAlgorithmclass::maximalSquare(vector<vector<char>>& matrix)
{
	//暴力法
	/*if (matrix.size() == 0) return 0;
	int ans = 0,row = matrix.size(), col = matrix[0].size();
	for (int i = 0; i < matrix.size(); ++i)
	{
		for (int j = 0; j < matrix[0].size(); j++)
		{
			if (matrix[i][j] == '1') {
				int sq = 1,r = i+1, c = j+1;
				bool sign = 1;
				while (r<row&&c<col)
				{
					for (int k = j; k <= c; ++k)
					{
						if (matrix[r][k] == '0') {
							sign = 0;
							break;
						}
					}
					for (int m = i; m <= r; ++m)
					{
						if (matrix[m][c] == '0') {
							sign = 0;
							break;
						}
					}
					r++;
					c++;
					if (sign)
						sq++;
				}
				ans = max(ans,sq);
			}
		}
	}
	return ans;*/
	//动态规划
	if (matrix.size() == 0) return 0;
	int ans = 0, row = matrix.size(), col = matrix[0].size();
	for (int i = 0; i < row; ++i)
	{
		if (matrix[i][0] == '1') {
			ans = 1;
			break;
		}
	}
	for (int i = 0; i < col; ++i)
	{
		if (matrix[0][i] == '1') {
			ans = 1;
			break;
		}
	}
	for (int i = 1; i < matrix.size(); ++i)
	{
		for (int j = 1; j < matrix[0].size(); j++)
		{
			if (matrix[i][j] == '0' || matrix[i - 1][j] == '0' || matrix[i][j - 1] == '0' || matrix[i - 1][j - 1] == '0') {
				ans = max(ans, matrix[i][j] - 48);
				continue;
			}
			matrix[i][j] = min(matrix[i - 1][j], min(matrix[i][j - 1], matrix[i - 1][j - 1])) + 1;
			ans = max(ans, matrix[i][j] - 48);
		}
	}
	return ans * ans;
}

int CAlgorithmclass::countNodes(TreeNode * root)
{
	int level = 0;
	int sSize = 0;
	queue<TreeNode*> q;
	q.push(root);
	while (!q.empty())
	{
		level++;
		sSize = q.size();
		for (int i = 0; i < sSize; ++i)
		{
			TreeNode* temp = q.front();
			q.pop();
			if (temp->left) q.push(temp->left);
			if (temp->right) q.push(temp->right);
		}
	}
	return pow(2, level - 1) - 1 + sSize;
	//递归
	/*if (!root) return 0;
	return countNodes(root->left) + countNodes(root->right) + 1;*/
}

int CAlgorithmclass::computeArea(int A, int B, int C, int D, int E, int F, int G, int H)
{
	int area1 = (C - A)*(D - B);
	int area2 = (G - E)*(H - F);
	if (C <= E || F >= D || B >= H || A >= G)
		return area1 + area2;
	int xmax = max(A, E), ymax = max(B, F);
	int xmin = min(G, C), ymin = min(D, H);
	return area1 - (xmin - xmax)*(ymin - ymax) + area2;
}

int CAlgorithmclass::numberOfSubarrays(vector<int>& nums, int k)
{
	vector<int> ptr;
	int ans = 0;
	ptr.push_back(-1);
	for (int i = 0; i < nums.size(); ++i) {
		if (nums[i] % 2 != 0)
			ptr.push_back(i);
	}
	ptr.push_back(nums.size());
	for (int i = 1; i + k < ptr.size(); ++i) {
		ans += (ptr[i] - ptr[i - 1])*(ptr[i + k] - ptr[i + k - 1]);
	}
	return ans;
}

int CAlgorithmclass::calculate(string s)
{
	stack<int> sk;
	stack<char> sc;
	for (int i = s.size() - 1; i >= 0; --i)
	{
		if (s[i] == ' ') continue;
		if (s[i] == '(') {
			while (sc.top() != ')')
			{
				int t = sk.top();
				sk.pop();
				if (sc.top() == '+') t += sk.top();
				else if (sc.top() == '-') t -= sk.top();
				sk.pop();
				sk.push(t);
				sc.pop();
			}
			sc.pop();
		}
		else if (s[i] >= 48 && s[i] <= 57) {
			int temp = i--;
			while (i >= 0 && s[i] >= 48 && s[i] <= 57) {
				i--;
			}
			int val = atoi(s.substr(i + 1, temp - i).c_str());
			sk.push(val);
			i++;
		}
		else {
			sc.push(s[i]);
		}

	}
	while (!sc.empty())
	{
		int t = sk.top();
		sk.pop();
		if (sc.top() == '+') t += sk.top();
		else if (sc.top() == '-') t -= sk.top();
		sk.pop();
		sk.push(t);
		sc.pop();
	}
	return sk.top();
}

TreeNode * CAlgorithmclass::invertTree(TreeNode * root)
{
	//递归
	/*if (!root) return root;
	TreeNode* right = invertTree(root->left);
	TreeNode* left = invertTree(root->right);
	root->right = right;
	root->left = left;
	return root;*/
	//迭代
	queue<TreeNode*> q;
	q.push(root);
	while (!q.empty())
	{
		TreeNode* temp = q.front();
		q.pop();
		TreeNode* sw = temp->left;
		temp->left = temp->right;
		temp->right = sw;
		if (temp->left) q.push(temp->left);
		if (temp->right) q.push(temp->right);
	}
	return root;
}

int CAlgorithmclass::calculate2(string s)
{
	stack<int> sint;
	stack<char> schar;
	schar.push('#');
	for (int i = 0; i < s.size(); ++i)
	{
		if (s[i] == ' ') continue;
		if (s[i] > 47 && s[i] < 58) {
			int temp = i++;
			while (i < s.size() && s[i] > 47 && s[i] < 58)
			{
				i++;
			}
			int val = atoi(s.substr(temp, i - temp).c_str());
			if (schar.top() == '-') sint.push(-val);
			else sint.push(val);
			i--;
			if (schar.top() == '*' || schar.top() == '/') {
				int t = sint.top();
				sint.pop();
				if (schar.top() == '*') t = sint.top()*t;
				else if (schar.top() == '/') t = sint.top() / t;
				sint.pop();
				sint.push(t);
				schar.pop();
			}
		}
		else
			schar.push(s[i]);
	}
	while (schar.top() != '#')
	{
		int t = sint.top();
		sint.pop();
		t += sint.top();
		sint.pop();
		sint.push(t);
		schar.pop();
	}
	return sint.top();
}

vector<string> CAlgorithmclass::summaryRanges(vector<int>& nums)
{
	if (nums.size() == 0) return{};
	vector<string> res;
	int l = nums[0], last = nums[0];
	for (int i = 1; i < nums.size(); ++i) {
		if (nums[i] != last + 1)
		{
			if (l == last) res.push_back(to_string(l));
			else res.push_back(to_string(l) + "->" + to_string(last));
			l = nums[i]; last = l;
		}
		else
			last = nums[i];
	}
	if (l == last) res.push_back(to_string(l));
	else res.push_back(to_string(l) + "->" + to_string(last));
	return res;
}

vector<int> CAlgorithmclass::majorityElement2(vector<int>& nums)
{
	vector<int> ans;
	int canditate1 = -1, canditate2 = -1, count1 = 0, count2 = 0;
	for (auto& num : nums)
	{
		if (num == canditate1)
			++count1;
		else if (num == canditate2)
			++count2;
		else if (count1 == 0) {
			canditate1 = num;
			count1 = 1;
		}
		else if (count2 == 0) {
			canditate2 = num;
			count2 = 1;
		}
		else {
			--count1;
			--count2;
		}
	}
	count1 = 0;
	count2 = 0;
	for (auto& x : nums) {
		if (x == canditate1) ++count1;
		else if (x == canditate2) ++count2;
	}
	if (count1 > nums.size() / 3) ans.push_back(canditate1);
	if (count2 > nums.size() / 3) ans.push_back(canditate2);
	return ans;
}

int CAlgorithmclass::kthSmallest(TreeNode * root, int k)
{
	//递归和迭代两种方法
	/*int num = 0;
	stack<TreeNode*> s;
	while (root||!s.empty())
	{
		while (root)
		{
			s.push(root);
			root = root->left;
		}
		root = s.top();
		s.pop();
		if (++num == k)
			return root->val;
		root = root->right;
	}
	return 0;*/
	//
	int leftn = findnumsTree(root->left);
	if (leftn + 1 == k) return root->val;
	else if (k <= leftn)
		return kthSmallest(root->left, k);
	else return kthSmallest(root->right, k);
}

bool CAlgorithmclass::isPowerOfTwo(int n)
{
	if (n <= 0) return 0;
	return (n&n - 1) == 0;
}

int CAlgorithmclass::countDigitOne(int n)
{
	//AC 50%
	/*if (n < 1) return 0;
	if (n < 9) return 1;
	string str = to_string(n);
	vector<int> bases(str.size());
	int base = 0,yin = 1;
	int ans = 0;
	for (int i = bases.size()-1; i >= 0; --i)
	{
		bases[i] = base;;
		base = base * 10 + yin;
		yin *= 10;
	}
	for (int i = 0; i < str.size()-1; ++i)
	{
		int temp = str[i] - 48;
		if (temp == 1) {
			int x = atoi(str.substr(i + 1).c_str());
			ans += bases[i] * (temp - 1) + x + 2;
			if (x == 0)
				ans--;
		}
		else {
			ans += yin / 10;
			ans += bases[i] * temp;
			if (str[i + 1] == 48)
				ans--;
		}
	}
	return ans;*/
	//分别计算n的个十百千位上的1的个数
	int num = n, ans = 0;
	long long i = 1;
	while (num) {
		if (num % 10 == 0)
			ans += (num / 10)*i;
		if (num % 10 == 1)
			ans += (num / 10)*i + (n%i) + 1;
		if (num % 10 > 1)
			ans += ceil(num / 10.0)*i;
		num /= 10;
		i *= 10;
	}
	return ans;
}

bool CAlgorithmclass::isPalindrome2(ListNode * head)
{
	//转换成数组
	/*if (!head) return 1;
	vector<int> res;
	while (head)
	{
		res.push_back(head->val);
		head = head->next;
	}
	int l = 0, r = res.size() - 1;
	while (l<r)
	{
		if (res[l++] != res[r--])
			return 0;
	}
	return 1;*/
	//快慢指针 反转 
	if (!head) return 1;
	ListNode* slow = head;
	ListNode* fast = head;
	while (fast&&fast->next)
	{
		slow = slow->next;
		fast = fast->next->next;
	}
	slow = reverse_list(slow);
	while (slow)
	{
		if (head->val != slow->val)
			return 0;
		head = head->next;
		slow = slow->next;
	}
	return 1;
}

TreeNode * CAlgorithmclass::lowestCommonAncestor(TreeNode * root, TreeNode * p, TreeNode * q)
{
	//多次重复搜索
	/*while (root)
	{
		if (root == p || root == q)
			return root;
		else if (!back_lowestCommonAncestor(root->left, p, q)) {
			root = root->right;
		}
		else if (!back_lowestCommonAncestor(root->right, p, q)) {
			root = root->left;
		}
		else
			break;
	}
	return root;*/
	//利用二叉搜索树的特点
	if (p->val == root->val || q->val == root->val)
		return root;
	if (p->val > root->val&&q->val > root->val)
		return lowestCommonAncestor(root->right, p, q);
	else if (p->val < root->val&&q->val < root->val)
		return lowestCommonAncestor(root->left, p, q);
	else
		return root;
}

TreeNode * CAlgorithmclass::lowestCommonAncestor2(TreeNode * root, TreeNode * p, TreeNode * q)
{
	if (!root || root == p || root == q) return root;
	TreeNode* left = lowestCommonAncestor2(root->left, p, q);
	TreeNode* right = lowestCommonAncestor2(root->right, p, q);
	if (left&&right) return root;
	if (left) return left;
	if (right) return right;
	return nullptr;
}

void CAlgorithmclass::deleteNode(ListNode * node)
{
	//逐个值传递
	/*ListNode* pre = nullptr;
	while (node&&node->next)
	{
		if (!node->next->next) {
			pre = node;
			break;
		}
		node->val = node->next->val;
		node = node->next;
	}
	node->val = node->next->val;
	node = node->next;
	pre->next = nullptr;
	delete node;*/
	//一步到位
	ListNode* temp = node->next;
	node->val = node->next->val;
	node->next = node->next->next;
	delete temp;
}

vector<int> CAlgorithmclass::productExceptSelf(vector<int>& nums)
{
	//左右互乘法
	/*vector<int> dp1(nums.size(),1),dp2(nums.size(),1);
	for (int i = 1; i < nums.size(); ++i)
	{
		dp1[i] = dp1[i - 1] * nums[i - 1];
	}
	for (int i = nums.size()-2; i >= 0; --i)
	{
		dp2[i] = dp2[i + 1] * nums[i + 1];
	}
	for (int i = 0; i < nums.size(); ++i)
	{
		dp1[i] = dp1[i] * dp2[i];
	}*/
	//优化空间
	vector<int> res(nums.size(), 1);
	int temp = 1;
	for (int i = 1; i < nums.size(); ++i)
	{
		res[i] = res[i - 1] * nums[i - 1];
	}
	for (int i = nums.size() - 1; i > 0; --i)
	{
		temp *= nums[i];
		res[i - 1] = res[i - 1] * temp;
	}
	return res;
}

vector<int> CAlgorithmclass::maxSlidingWindow(vector<int>& nums, int k)
{
	//哈希表
	/*queue<int> q;
	map<int,int> amap;
	vector<int> ans;
	for (int i = 0; i < nums.size(); ++i)
	{
		q.push(nums[i]);
		amap[nums[i]]++;
		if (q.size() >= k) {
			pair<int, int> temp = *--amap.end();
			int f = q.front();
			q.pop();
			if (amap[f] > 1)
				amap[f]--;
			else
				amap.erase(f);
			ans.push_back(temp.first);
		}
	}
	return ans;*/
	//双向队列
	if (k == 0) return{};
	vector<int> res;
	deque<int> deque;
	for (int i = 0; i < nums.size(); i++) {
		if (!deque.empty() && deque.front() <= i - k)
			deque.pop_front();
		while (!deque.empty() && nums[i] > nums[deque.back()])
			deque.pop_back();
		deque.push_back(i);
		if (i >= k - 1) res.push_back(nums[deque.front()]);
	}
	return res;
}

int CAlgorithmclass::waysToChange(int n)
{
	//数学
	/*int ans = 0, mod = 1000000007;
	for (int i = 0; i*25 <= n; ++i)
	{
		int r = n - 25 * i;
		int a = r / 10;
		int b = r % 10 / 5;
		ans = (ans + (long long)(a + 1) * (a + b + 1) % mod) % mod;
	}
	return ans;*/
	//动态规划 背包问题
	int mod = 1000000007;
	vector<int> f(n + 1, 0), c;
	f[0] = 1;
	c = { 25,10,5,1 };
	for (int i = 0; i < c.size(); ++i) {
		for (int j = c[i]; j <= n; ++j)
			f[j] = (f[j] + f[j - c[i]]) % mod;
	}
	return f[n];
}

bool CAlgorithmclass::searchMatrix2(vector<vector<int>>& matrix, int target)
{
	if (matrix.size() == 0) return false;
	int i = 0, j = matrix[0].size() - 1;
	while (i < matrix.size() && j >= 0) {
		if (matrix[i][j] == target)  return true;
		else if (matrix[i][j] < target) ++i;
		else  --j;
	}
	return false;
}

int CAlgorithmclass::reversePairs(vector<int>& nums)
{
	vector<int> temp(nums.size());
	return mergeSort(nums, temp, 0, nums.size() - 1);
}

vector<int> CAlgorithmclass::diffWaysToCompute(string input)
{
	if (m_svmap.find(input) != m_svmap.end())
		return m_svmap[input];
	vector<int> res;
	for (int i = 0; i < input.size(); ++i)
	{
		if (input[i] == '+' || input[i] == '-' || input[i] == '*') {
			vector<int> res1 = diffWaysToCompute(input.substr(0, i));
			vector<int> res2 = diffWaysToCompute(input.substr(i + 1));
			for (auto r1 : res1)
			{
				for (auto r2 : res2)
				{
					if (input[i] == '+') {
						res.push_back(r1 + r2);
					}
					else if (input[i] == '-') {
						res.push_back(r1 - r2);
					}
					else {
						res.push_back(r1 * r2);
					}
				}
			}
		}
	}
	if (res.empty())
		res.push_back(stoi(input));
	m_svmap[input] = res;
	return res;
}

bool CAlgorithmclass::isAnagram(string s, string t)
{
	//哈希表
	/*if (s.size() != t.size())
		return false;
	unordered_map<char,int> smap;
	for (int i = 0; i < s.size(); ++i)
	{
		smap[s[i]]++;
	}
	for (int i = 0; i < t.size(); ++i)
	{
		if (smap.find(t[i]) == smap.end()||smap[t[i]] == 0)
			return false;
		smap[t[i]]--;
	}
	return true;*/
	//数组
	if (s.size() != t.size())
		return false;
	vector<int> temp(26, 0);
	for (int i = 0; i < s.size(); ++i)
	{
		temp[s[i] - 'a']++;
	}
	for (int i = 0; i < t.size(); ++i)
	{
		int x = t[i] - 'a';
		if (temp[x] == 0)
			return false;
		temp[x]--;
	}
	return true;
}

vector<string> CAlgorithmclass::binaryTreePaths(TreeNode * root)
{
	if (!root) return{};
	string s = "";
	back_binaryTreePaths(root, s);
	return m_strVt;
}

int CAlgorithmclass::addDigits(int num)
{
	/*int ans = num;
	while (ans>9)
	{
		ans = 0;
		while (num)
		{
			int temp = num % 10;
			num /= 10;
			ans += temp;
		}
		num = ans;
	}
	return ans;*/
	//数学方法
	int temp = num % 9;
	return temp == 0 ? 9 : temp;
}

vector<vector<int>> CAlgorithmclass::permute(vector<int>& nums)
{
	vector<int> temp;
	vector<bool> visit(nums.size(), false);
	back_permute(nums, temp, visit);
	return m_intVtVt;
}

bool CAlgorithmclass::isUgly(int num)
{
	int temp[3] = { 2,3,5 };
	for (int i = 0; i < 3; ++i)
	{
		while (num > 1)
		{
			if (num%temp[i] != 0) break;
			num /= temp[i];
		}
	}
	return num == 1 ? true : false;
}

int CAlgorithmclass::nthUglyNumber(int n)
{
	vector<int> ans = { 1 };
	int ptr2 = 0, ptr3 = 0, ptr5 = 0;
	while (ans.size() < n)
	{
		int a = ans[ptr2] * 2, b = ans[ptr3] * 3, c = ans[ptr5] * 5;
		int imin = min(a, min(b, c));
		ans.push_back(imin);
		if (imin == a) ++ptr2;
		if (imin == b) ++ptr3;
		if (imin == c) ++ptr5;
	}
	return ans[n - 1];
}

int CAlgorithmclass::missingNumber(vector<int>& nums)
{
	// 数学 求和
	/*int n = nums.size();
	int sum = (n + 1)*n / 2;
	for (auto&i : nums) {
		sum -= i;
	}
	return sum;*/
	// 数学 边加边减  防止溢出
	/*int sum = 0, i;
	for (i = 0; i < nums.size(); ++i) {
		sum += nums[i] - i;
	}
	return abs(sum - i);*/
	//位运算
	int ans = 0;
	for (int i = 0; i < nums.size(); ++i)
	{
		ans = ans ^ nums[i] ^ i;
	}
	return ans ^ nums.size();
}

int CAlgorithmclass::movingCount(int m, int n, int k)
{
	if (m == 0 || n == 0) return 1;
	m_int = 1;
	m_pairVt = { {1,0},{-1,0},{0,1},{0,-1} };
	vector<vector<bool>> visit(m, vector<bool>(n, false));
	visit[0][0] = true;
	back_movingCount(visit, 0, 0, k);
	return m_int;
}

void CAlgorithmclass::rotate(vector<vector<int>>& matrix)
{
	//大风车转
	/*int row = matrix.size()-1, col = matrix[0].size()-1,temp = col;
	for (int i = 0; i < matrix.size()/2; ++i)
	{
		for (int j = i; j < temp; ++j)
		{
			swap(matrix[i][j], matrix[j][col - i]);
			swap(matrix[i][j], matrix[row - j][i]);
			swap(matrix[row - j][i], matrix[row - i][col - j]);
		}
		temp -= 1;
	}*/
	//先水平翻转，再主对角线翻转
	int n = matrix.size();
	// 水平翻转
	for (int i = 0; i < n / 2; ++i) {
		for (int j = 0; j < n; ++j) {
			swap(matrix[i][j], matrix[n - i - 1][j]);
		}
	}
	// 主对角线翻转
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < i; ++j) {
			swap(matrix[i][j], matrix[j][i]);
		}
	}
}

void CAlgorithmclass::gameOfLife(vector<vector<int>>& board)
{
	int row = board.size(), col = board[0].size();
	m_pairVt = { {-1,-1},{-1,0},{-1,1},{0,1},{1,1},{1,0},{1,-1},{0,-1} };
	//vector<vector<int>> cop = board;
	for (int i = 0; i < board.size(); ++i)
	{
		for (int j = 0; j < board[0].size(); ++j)
		{
			int num = 0;
			for (int k = 0; k < 8; ++k)
			{
				int temp_i = i + m_pairVt[k].first;
				int temp_j = j + m_pairVt[k].second;
				if (temp_i < 0 || temp_j < 0 || temp_i >= row || temp_j >= col)
					continue;
				if (abs(board[temp_i][temp_j]) == 1)
					++num;
			}
			if (board[i][j] == 1 && (num < 2 || num>3))
				board[i][j] = -1;
			else if (num == 3)
				board[i][j] = 2;
		}
	}
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			if (board[row][col] > 0)
				board[row][col] = 1;
			else
				board[row][col] = 0;
		}
	}
}

ListNode * CAlgorithmclass::mergeKLists(vector<ListNode*>& lists)
{
	//第一种分治
	/*if (lists.empty()) return nullptr;
	int sSize = lists.size();
	while (sSize>1)
	{
		int i = 0,start = i, end = sSize - i - 1;
		while (start<end)
		{
			ListNode* s = lists[start];
			ListNode* e = lists[end];
			ListNode* head = new ListNode(0);
			ListNode* cop = head;
			while (s&&e)
			{
				if (s->val < e->val) {
					head->next = s;
					s = s->next;
				}
				else {
					head->next = e;
					e = e->next;
				}
				head = head->next;
			}
			head->next = s == nullptr ? e : s;
			lists[start] = cop->next;
			cop->next = nullptr;
			delete cop;
			start++;
			end--;
		}
		sSize = (sSize + 1) / 2;
	}
	return lists[0];*/
	//第二种分治
	/*return mergeLists(lists, 0, lists.size() - 1);*/
	//堆
	priority_queue<status> q;
	for (int i = 0; i < lists.size(); ++i)
		if (lists[i]) q.push({ lists[i]->val,lists[i] });
	ListNode* head = new ListNode(0);
	ListNode* cop = head;
	while (!q.empty())
	{
		status temp = q.top();
		q.pop();
		head->next = temp.ptr;
		head = head->next;
		if (temp.ptr->next) q.push({ temp.ptr->next->val,temp.ptr->next });
	}
	head = cop->next;
	cop->next = nullptr;
	delete cop;
	return head;
}

vector<int> CAlgorithmclass::maxDepthAfterSplit(string seq)
{
	stack<int> a, b;
	vector<int> ans(seq.size(), 0);
	for (int i = 0; i < seq.size(); ++i)
	{
		if (seq[i] == '(' && (a.empty() || a.size() < b.size()))
			a.push(seq[i]);
		else if (seq[i] == '(' && (b.empty() || a.size() >= b.size())) {
			b.push(seq[i]);
			ans[i] = 1;
		}
		else {
			if (a.size() < b.size()) {
				b.pop();
				ans[i] = 1;
			}
			else a.pop();
		}
	}
	return ans;
}

int CAlgorithmclass::search2(vector<int>& nums, int target)
{
	if (nums.size() == 0) return -1;
	int l = 0, r = nums.size() - 1;
	while (l <= r)
	{
		int mid = l + (r - l) / 2;
		if (nums[mid] == target)
			return mid;
		if (nums[mid] >= nums[l]) {
			if (nums[l] <= target && target < nums[mid])
				r = mid - 1;
			else
				l = mid + 1;
		}
		else {
			if (nums[mid] < target&&target <= nums[r])
				l = mid + 1;
			else
				r = mid - 1;
		}
	}
	return (l < nums.size() && nums[l] == target) ? l : -1;
}

int CAlgorithmclass::superEggDrop(int K, int N)
{
	int t = 1;
	while (back_calculateF(K, t) < N + 1)
		t++;
	return t;
}

int CAlgorithmclass::hIndex(vector<int>& citations)
{
	//排序
	/*int ans = 0, end = citations.size();
	sort(citations.begin(),citations.end());
	for (int i = 0; i < citations.size(); ++i)
	{
		if (citations[i] == (end - i))
			ans = max(ans,i);
	}
	return ans;*/
	//计数排序
	int n = citations.size();
	vector<int> cals(n + 1, 0);
	for (auto& i : citations)
		cals[min(n, i)]++;
	int k = n;
	for (int s = cals[n]; k > s; s += cals[k])
		k--;
	return k;
}

int CAlgorithmclass::hIndex2(vector<int>& citations)
{
	if (citations.size() == 0) return 0;
	int end = citations.size(), l = 0, r = citations.size() - 1;
	while (l < r)
	{
		int mid = l + (r - l) / 2;
		if (citations[mid] >= (end - mid))
			r = mid;
		else
			l = mid + 1;
	}
	return citations[l] >= end - l ? end - l : 0;
}

int CAlgorithmclass::numSquares(int n)
{
	//广度优先搜索
	/*vector<int> dp;
	for (int i = 1; i*i <= n; ++i)
		dp.push_back(i*i);
	int ans = 0;
	queue<int> q;
	q.push(n);
	while (!q.empty())
	{
		ans++;
		int sSize = q.size();
		for (int i = 0; i < sSize; ++i)
		{
			int temp = q.front();
			q.pop();
			for (int j = 0; j < dp.size(); ++j)
			{
				if (temp < dp[j])
					break;
				int val = temp - dp[j];
				if (val == 0)
					return ans;
				q.push(temp - dp[j]);
			}
		}
	}
	return 0;*/
	//动态规划
	/*vector<int> dp(n+1,INT_MAX),sq;
	dp[0] = 0;
	for (int i = 1; i*i <= n; ++i)
		sq.push_back(i*i);
	for (int i = 1; i <= n; ++i)
	{
		for (int j = 0; j < sq.size(); ++j)
		{
			if (i < sq[j])
				break;
			dp[i] = min(dp[i],dp[i - sq[j]] + 1);
		}
	}
	return dp[n];*/
	//四平方定理： 任何一个正整数都可以表示成不超过四个整数的平方之和
	//推论：满足四数平方和定理的数n（四个整数的情况），必定满足 n=4^a(8b+7)
	int num = n;
	while (n % 4 == 0)
	{
		n /= 4;
	}
	if (n % 8 == 7)
		return 4;
	int sq = pow(n, 0.5);
	if (n == sq * sq)
		return 1;
	for (int i = 1; i*i <= n; ++i)
	{
		int x = n - i * i;
		int sq = pow(x, 0.5);
		if (x == sq * sq)
			return 2;
	}
	return 3;
}

vector<string> CAlgorithmclass::addOperators(string num, int target)
{
	//回溯
	if (num == "2147483648"&&target == INT_MIN) return{};
	string resstr = "";
	back_addOperators(num, 0, resstr, 0, 1, target);
	return m_strVt;
}

vector<int> CAlgorithmclass::singleNumbers(vector<int>& nums)
{
	vector<int> ans(2, 0);
	for (auto& i : nums)
		ans[0] ^= i;
	int temp = ans[0] & (-ans[0]);
	for (auto& i : nums) {
		if (i&temp)
			ans[1] ^= i;
	}
	ans[0] ^= ans[1];
	return ans;
}

void CAlgorithmclass::moveZeroes(vector<int>& nums)
{
	/*int zero = 0,sSize = nums.size();
	for (int i = 0; i < sSize; ++i)
	{
		if (nums[i] == 0)
			zero++;
		else
			nums[i - zero] = nums[i];
	}
	for (int i = sSize -1; i >= sSize - zero; --i)
		nums[i] = 0;*/
		//优化
	for (int i = 0, j = 0; j < nums.size(); ++j)
	{
		if (nums[j] != 0)
			swap(nums[i++], nums[j]);
	}
}

int CAlgorithmclass::findDuplicate(vector<int>& nums)
{
	int slow = 0, fast = 0;
	while (slow != fast && fast != 0)
	{
		slow = nums[slow];
		fast = nums[nums[fast]];
	}
	for (int i = 0; slow != i; i = nums[i])
	{
		slow = nums[slow];
	}
	return slow;
}

bool CAlgorithmclass::wordPattern(string pattern, string str)
{
	/*int k = 0;
	for (int i = 0; i < str.size(); ++i)
		if (str[i] == ' ') k++;
	if (k + 1 != pattern.size()) return false;
	unordered_map<char, string> amap;
	unordered_map<string, char> bmap;
	int l = 0;
	for (int i = 0; i < pattern.size(); ++i)
	{
		int r = l;
		while (r<str.size()&&str[r] != ' ')
			r++;
		string s = str.substr(l,r - l);
		l = r + 1;
		if (amap.find(pattern[i]) != amap.end() && amap[pattern[i]] != s)
			return false;
		if (bmap.find(s) != bmap.end() && bmap[s] != pattern[i])
			return false;
		amap[pattern[i]] = s;
		bmap[s] = pattern[i];
	}
	return true;*/
	//先分割字符串,再匹配
	vector<string> v;
	string c = " ";
	SplitString(pattern, c, v);
	if (v.size() != pattern.size()) return false;
	unordered_map<char, string> amap;
	unordered_map<string, char> bmap;
	for (int i = 0; i < pattern.size(); ++i)
	{
		if (amap.find(pattern[i]) != amap.end() && amap[pattern[i]] != v[i])
			return false;
		if (bmap.find(v[i]) != bmap.end() && bmap[v[i]] != pattern[i])
			return false;
		amap[pattern[i]] = v[i];
		bmap[v[i]] = pattern[i];
	}
	return true;
}

bool CAlgorithmclass::canWinNim(int n)
{
	return n % 4 == 0 ? false : true;
}

int CAlgorithmclass::lengthOfLIS(vector<int>& nums)
{
	//动态规划
	/*int ans = 1;
	vector<int> dp(nums.size(), 1);
	for (int i = 0; i < nums.size(); ++i)
	{
		for (int j = i - 1; j >= 0; --j)
		{
			if (nums[j] < nums[i]) {
				dp[i] = max(dp[i],dp[j]+1);
			}
		}
		ans = max(ans,dp[i]);
	}
	return ans;*/
	//动态规划+二分查找
	int ans = 0;
	vector<int> tail(nums.size(), 0);
	for (int i = 0; i < nums.size(); ++i)
	{
		int l = 0, r = ans;
		while (l < r)
		{
			int mid = l + (r - l) / 2;
			if (tail[mid] < nums[i]) l = mid + 1;
			else r = mid;
		}
		tail[l] = nums[i];
		if (ans == r) ans++;
	}
	return ans;
}

int CAlgorithmclass::coinChange(vector<int>& coins, int amount)
{
	//动态规划
	/*int imax = amount + 1;
	vector<int> dp(amount + 1, imax);
	dp[0] = 0;
	for (int i = 1; i <= amount; ++i)
	{
		for (int j = 0; j < coins.size(); ++j)
		{
			if (coins[j] <= i)
				dp[i] = min(dp[i], dp[i - coins[j]] + 1);
		}
	}
	return dp[amount]>amount?-1:dp[amount];*/
	//递归
	sort(coins.begin(), coins.end());
	m_int = amount + 1;
	back_coinChange(coins, coins.size() - 1, amount, 0);
	return m_int > amount ? -1 : m_int;
}

bool CAlgorithmclass::canMeasureWater(int x, int y, int z)
{
	if (x + y < z) return false;
	if (x == 0 || y == 0) return z == 0 || x + y == z;
	while (x%y != 0)
	{
		int temp = y;
		y = x % y;
		x = temp;
	}
	return z % y == 0;
	//return z%gcd(x,y) == 0;
}

int CAlgorithmclass::findInMountainArray(int target, vector<int>& nums)
{
	int l = 0, r = nums.size() - 1, ans = INT_MAX;
	while (l < r) {
		int mid = l + (r - l) / 2;
		int mid_val = nums[mid];
		if (mid_val < nums[mid + 1]) l = mid + 1;
		else r = mid;
	}
	int m = l;
	r = m, l = 0;
	while (l <= r) {
		int mid = l + (r - l) / 2;
		int mid_val = nums[mid];
		if (mid_val == target) return mid;
		if (mid_val < target)	l = mid + 1;
		else r = mid - 1;
	}
	ans = nums[l] == target ? l : ans;
	l = m + 1, r = nums.size() - 1;
	while (l <= r) {
		int mid = l + (r - l) / 2;
		int mid_val = nums[mid];
		if (mid_val == target) return mid;
		if (mid_val > target) l = mid + 1;
		else r = mid - 1;
	}
	return (l < nums.size() && target == nums[l]) ? l : -1;
}

int CAlgorithmclass::longestPalindrome(string s)
{
	int ans = 0;
	unordered_map<char, int> amap;
	for (int i = 0; i < s.size(); ++i)
	{
		if (amap.find(s[i]) != amap.end()) {
			if (++amap[s[i]] % 2 == 0)
				ans += 2;
		}
		else amap[s[i]]++;
	}
	if (amap.size() == 1) return s.size() % 2 == 0 ? s.size() / 2 : s.size() / 2 + 1;
	for (unordered_map<char, int>::iterator i = amap.begin(); i != amap.end(); ++i) {
		if (i->second == 1) {
			ans++;
			break;
		}
	}
	return ans;
}

int CAlgorithmclass::diameterOfBinaryTree(TreeNode * root)
{
	if (!root) return 0;
	back_diameterOfBinaryTree(root);
	return m_int;
}

int CAlgorithmclass::maxAreaOfIsland(vector<vector<int>>& grid)
{
	m_pairVt = { {-1,0},{0,1},{1,0},{0,-1} };
	m_row = grid.size(), m_col = grid[0].size();
	for (int i = 0; i < m_row; ++i)
	{
		for (int j = 0; j < m_col; ++j)
		{
			if (grid[i][j] == 1) {
				int a = 1;//需要把a也替换成全局变量
				back_maxAreaOfIsland(grid, i, j);
				m_int = max(m_int, a);
			}
		}
	}
	return m_int;
}

int CAlgorithmclass::minimumLengthEncoding(vector<string>& words)
{
	//反转
	/*for (auto &s : words) {
		reverse(s.begin(), s.end());
	}
	sort(words.begin(), words.end());
	int res = 0;
	for (int i = 0; i<words.size() - 1; i++) {
		int size = words[i].size();
		if (words[i] == words[i + 1].substr(0, size))
			continue;
		res += size + 1;
	}
	return res + words.back().size() + 1;*/
	//哈希表
	/*unordered_set<string> good(words.begin(), words.end());
	for (const string& word : words) {
		for (int k = 1; k < word.size(); ++k) {
			good.erase(word.substr(k));
		}
	}
	int ans = 0;
	for (const string& word : good) {
		ans += word.size() + 1;
	}
	return ans;*/
	//字典树
	Trie2* root = new Trie2();
	int ans = 0;
	for (auto& word : words) {
		ans += root->insert(word);
	}
	return ans;
}

bool CAlgorithmclass::isRectangleOverlap(vector<int>& rec1, vector<int>& rec2)
{
	return !(rec2[0] >= rec1[2] || rec1[0] >= rec2[2] || rec2[1] >= rec1[3] || rec1[1] >= rec2[3]);
}

int CAlgorithmclass::surfaceArea(vector<vector<int>>& grid)
{
	if (grid.size() == 0) return 0;
	int ans = 0, row = grid.size(), col = grid[0].size();
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			if (grid[i][j] < 1) continue;
			ans += (grid[i][j] << 2) + 2;
			if (i - 1 >= 0) ans -= min(grid[i][j], grid[i - 1][j]) << 1;
			if (j - 1 >= 0) ans -= min(grid[i][j], grid[i][j - 1]) << 1;
		}
	}
	return ans;
}

bool CAlgorithmclass::hasGroupsSizeX(vector<int>& deck)
{
	int counts[10000] = { 0 };
	int g = 0;
	for (int d : deck) {
		counts[d]++;
	}
	for (int b : counts) {
		if (b == 0) continue;
		g = gcd(b, g);
		if (g == 1) return false;
	}
	return g >= 2;
}

int CAlgorithmclass::minIncrementForUnique(vector<int>& A)
{
	int n[80000] = { 0 };
	int len = A.size();
	for (int i = 0; i < len; i++)
		n[A[i]]++;
	int move = 0;
	for (int i = 0; i < 80000; i++) {
		if (n[i] <= 1)
			continue;
		move += n[i] - 1;
		n[i + 1] += n[i] - 1;
		n[i] = 1;
	}
	return move;
}

int CAlgorithmclass::orangesRotting(vector<vector<int>>& grid)
{
	if (!grid.size() || !grid[0].size()) return 0;
	int ans = 0, row = grid.size(), col = grid[0].size(), fresh = 0;
	m_pairVt = { {-1,0},{0,1},{1,0},{0,-1} };
	queue<pair<int, int>> q;
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			if (grid[i][j] == 2)
				q.push(make_pair(i, j));
			else if (grid[i][j] == 1)
				++fresh;
		}
	}
	if (!fresh) return 0;
	while (!q.empty())
	{
		ans++;
		int sSize = q.size();
		for (int i = 0; i < sSize; ++i)
		{
			pair<int, int> temp = q.front();
			q.pop();
			for (int k = 0; k < 4; ++k)
			{
				int temp_i = temp.first + m_pairVt[k].first;
				int temp_j = temp.second + m_pairVt[k].second;
				if (temp_i < 0 || temp_j < 0 || temp_i >= row || temp_j >= col || grid[temp_i][temp_j] != 1)
					continue;
				fresh--;
				grid[temp_i][temp_j] = 2;
				q.push(make_pair(temp_i, temp_j));
			}
		}
	}
	return fresh ? -1 : ans;
}

int CAlgorithmclass::numRookCaptures(vector<vector<char>>& board)
{
	int ri, rj, ans = 0;
	m_pairVt = { { -1,0 },{ 0,1 },{ 1,0 },{ 0,-1 } };
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 8; ++j)
		{
			if (board[i][j] == 'R') {
				ri = i;
				rj = j;
				break;
			}
		}
	}
	for (int i = 0; i < 4; ++i)
	{
		int temp_i = ri + m_pairVt[i].first, temp_j = rj + m_pairVt[i].second;
		while (temp_i >= 0 && temp_j >= 0 && temp_i < 8 && temp_j < 8)
		{
			if (board[temp_i][temp_j] != '.') {
				if (islower(board[temp_i][temp_j]))
					ans++;
				break;
			}
			temp_i = temp_i + m_pairVt[i].first;
			temp_j = temp_j + m_pairVt[i].second;
		}
	}
	return ans;
}

bool CAlgorithmclass::canThreePartsEqualSum(vector<int>& A)
{
	int sum = 0;
	for (auto& a : A)
		sum += a;
	if (sum % 3) return false;
	sum /= 3;
	int i = 1, j = A.size() - 2, sumi = A[0], sumj = A[A.size() - 1];
	while (i < A.size() && j >= 0 && i < j)
	{
		if (sumi != sum)
			sumi += A[i++];
		if (sumj != sum)
			sumj += A[j--];
		if (sum == sumi && sum == sumj)
			break;
	}
	return (i <= j && sum == sumi && sum == sumj) ? true : false;
}

string CAlgorithmclass::gcdOfStrings(string str1, string str2)
{
	//辗转相减
	/*if (str1 == "" || str2 == "") return "";
	string ans = "";
	while (str1.size()!= str2.size())
	{
		if (str1.size() < str2.size())
			swap(str1,str2);
		str1 = str1.substr(str2.size());
	}
	return str1 == str2 ? str1 : "";*/
	//辗转相除  最大公约数
	if (str1 + str2 != str2 + str1) return "";
	return str1.substr(0, gcd((int)str1.length(), (int)str2.length()));
}

vector<int> CAlgorithmclass::distributeCandies(int candies, int num_people)
{
	vector<int> ans(num_people, 0);
	int i = 0;
	while (candies > 0)
	{
		int temp = min(candies, i + 1);
		ans[i % num_people] += temp;
		candies -= temp;
		++i;
	}
	return ans;
}

int CAlgorithmclass::countCharacters(vector<string>& words, string chars)
{
	//哈希
	/*int ans = 0;
	unordered_map<char, int> amap;
	for (auto& ch : chars)
		amap[ch]++;
	for (auto& word : words) {
		unordered_map<char, int> temp_map = amap;
		int i = 0;
		for (; i < word.size();++i) {
			char ch = word[i];
			if (temp_map.find(ch) == temp_map.end()) break;
			temp_map[ch]--;
			if (temp_map[ch] == 0) temp_map.erase(ch);
		}
		if (i == word.size()) ans += word.size();
	}
	return ans;*/
	//用数组映射
	int ans = 0;
	int issign[125] = { 0 };
	for (auto& ch : chars)
		issign[ch]++;
	for (auto& word : words) {
		int i = 0;
		int sign[125];
		memcpy(sign, issign, sizeof(issign));
		for (; i < word.size(); ++i) {
			char ch = word[i];
			if (!sign[ch]) break;
			sign[ch]--;
		}
		if (i == word.size()) ans += word.size();
	}
	return ans;
}

int CAlgorithmclass::maxDistance(vector<vector<int>>& grid)
{
	int N = grid.size();
	m_pairVt = { {-1,0},{0,1},{1,0},{0,-1} };
	queue<pair<int, int>> q;
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			if (grid[i][j])
				q.push(make_pair(i, j));
		}
	}
	bool isSingle = false;
	pair<int, int> temp = { 0,0 };
	while (!q.empty())
	{
		temp = q.front();
		q.pop();
		for (int k = 0; k < 4; ++k) {
			int temp_i = temp.first + m_pairVt[k].first;
			int temp_j = temp.second + m_pairVt[k].second;
			if (temp_i < 0 || temp_j < 0 || temp_i >= N || temp_j >= N || grid[temp_i][temp_j] != 0)
				continue;
			grid[temp_i][temp_j] = grid[temp.first][temp.second] + 1;
			q.push(make_pair(temp_i, temp_j));
			isSingle = true;
		}
	}
	return isSingle ? grid[temp.first][temp.second] - 1 : -1;
}

string CAlgorithmclass::compressString(string S)
{
	if (!S.size()) return "";
	string ans = S.substr(0, 1);
	int count = 1;
	for (int i = 1; i < S.size(); ++i) {
		if (S[i] == S[i - 1]) {
			count++;
		}
		else {
			ans += to_string(count) + S[i];
			count = 1;
		}
	}
	ans += to_string(count);
	return ans.size() == S.size() ? S : ans;
}

int CAlgorithmclass::massage(vector<int>& nums)
{
	//动态规划
	int pre = 0, cur = 0;//pre表示dp[i-1]：前i-1个最大值   cur表示dp[i]：前i个最大值
	for (auto& num : nums) {
		int temp = cur;
		cur = max(cur, pre + num);
		pre = temp;
	}
	return cur;
}

vector<vector<int>> CAlgorithmclass::findContinuousSequence(int target)
{
	//双向队列
	/*vector<vector<int>> ans;
	deque<int> q;
	for (int i = 1; i <= target/2; ++i)
	{
		q.push_back(i);
		int sum = (i + q.front())*q.size() / 2;
		if (sum == target&&q.size() != 1)
			ans.push_back(vector<int>(q.begin(), q.end()));
		while (sum > target)
		{
			sum -= q.front();
			q.pop_front();
			if (sum == target&&q.size() != 1)
				ans.push_back(vector<int>(q.begin(), q.end()));
		}
	}
	return ans;*/
	//双指针
	vector<vector<int>>vec;
	vector<int> res;
	for (int l = 1, r = 2; l < r;) {
		int sum = (l + r) * (r - l + 1) / 2;
		if (sum == target) {
			res.clear();
			for (int i = l; i <= r; ++i) res.emplace_back(i);
			vec.emplace_back(res);
			l++;
		}
		else if (sum < target) r++;
		else l++;
	}
	return vec;
}

int CAlgorithmclass::lastRemaining(int n, int m)
{
	int ans = 0;
	for (int i = 2; i <= n; ++i)
		ans = (ans + m) % 2;
	return ans;
}

int CAlgorithmclass::jump(vector<int>& nums)
{
	/*int i = 1,ans = 0,start = 0;
	while (i<nums.size())
	{
		int step = 0;
		for (int j = start; j < i; ++j)
			step = max(step, nums[j] + j);
		ans++;
		start = i;
		i = step + 1;
	}
	return ans;*/
	//优化
	int ans = 0, end = 0, maxPos = 0;
	for (int i = 0; i < nums.size() - 1; i++)
	{
		maxPos = max(nums[i] + i, maxPos);
		if (i == end)
		{
			end = maxPos;
			ans++;
		}
	}
	return ans;
}

string CAlgorithmclass::getHint(string secret, string guess)
{
	int dp[125];
	int r = 0, w = 0;
	for (int i = 0; i < secret.size(); ++i)
	{
		if (secret[i] == guess[i])
			r++;
		else
			dp[secret[i]]++;
	}
	for (int i = 0; i < guess.size(); ++i)
	{
		if (secret[i] != guess[i] && dp[guess[i]] != 0) {
			dp[guess[i]]--;
			w++;
		}
	}
	return to_string(r) + 'A' + to_string(w) + 'B';
}

vector<string> CAlgorithmclass::removeInvalidParentheses(string s)
{
	unordered_set<string> avisited;
	queue<string> q;
	avisited.insert(s);
	q.push(s);
	vector<string> ans;
	while (!q.empty())
	{
		int sSize = q.size();
		if (!ans.empty()) break;
		for (int i = 0; i < sSize; ++i)
		{
			string curstr = q.front();
			q.pop();
			if (isValidParentheses(curstr))
				ans.push_back(curstr);
			for (int j = 0; j < curstr.size(); ++j)
			{
				if (curstr[j] != '('&&curstr[j] != ')') continue;
				string newstr = curstr.substr(0, j) + curstr.substr(j + 1);
				if (avisited.find(newstr) == avisited.end()) {
					avisited.insert(newstr);
					q.push(newstr);
				}
			}
		}
	}
	return ans;
}

bool CAlgorithmclass::isValidParentheses(string s)
{
	if (s.empty()) return true;
	int count = 0;
	for (auto& c : s) {
		if (c == '(')
			count++;
		else if (c == ')')
			count--;
		if (count < 0) return false;
	}
	return count == 0;
}

int CAlgorithmclass::mincostTickets(vector<int>& days, vector<int>& costs)
{
	int n = days[days.size() - 1];
	vector<int> dp(n + 1, 0);
	int a, b, c;
	for (auto& i : days)
		dp[i] = -1;
	for (int i = 1; i <= n; ++i)
	{
		if (dp[i] == 0) {
			dp[i] = dp[i - 1];
			continue;
		}
		a = dp[i - 1] + costs[0];
		b = (i >= 7 ? dp[i - 7] : 0) + costs[1];
		c = (i >= 30 ? dp[i - 30] : 0) + costs[2];
		dp[i] = min({ a,b,c });
	}
	return dp[n];
}

bool CAlgorithmclass::isAdditiveNumber(string num)
{
	vector<long double>tmp;
	return back_isAdditiveNumber(num, tmp);
}

bool CAlgorithmclass::isSubtree(TreeNode * s, TreeNode * t)
{
	//递归
	/*if (s == nullptr)
		return false;
	if (s->val == t->val&& back_isSubtree(s, t)) {
		return true;
	}
	if (isSubtree(s->left, t) || isSubtree(s->right, t))
		return true;
	return false;*/
	//字符串kmp算法
	return false;
}

vector<int> CAlgorithmclass::findMinHeightTrees(int n, vector<vector<int>>& edges)
{
	//暴力解法 遍历每个根节点的高度
	/*vector<int> ans;
	vector<int> indegrees_ori(n,0);
	vector<vector<int>> adjacency(n);
	for (int i = 0; i < edges.size(); ++i) {
		indegrees_ori[edges[i][0]]++;
		indegrees_ori[edges[i][1]]++;
		adjacency[edges[i][0]].push_back(edges[i][1]);
		adjacency[edges[i][1]].push_back(edges[i][0]);
	}
	vector<int> res(n,0);
	for (int i = 0; i < n; ++i)
	{
		int level = 0;
		vector<int> indegrees = indegrees_ori;
		queue<int> q;
		q.push(i);
		while (!q.empty())
		{
			++level;
			int sSize = q.size();
			for (int k = 0; k < sSize; ++k) {
				int temp = q.front();
				q.pop();
				for (int j = 0; j < adjacency[temp].size(); ++j) {
					if (indegrees[adjacency[temp][j]] > 0) {
						q.push(adjacency[temp][j]);
						--indegrees[temp];
						--indegrees[adjacency[temp][j]];
					}
				}
			}
		}
		res[i] = level;
	}
	int minlevel = n + 1;
	for (int i = 0; i < n; ++i)
		minlevel = min(res[i],minlevel);
	for (int i = 0; i < n; ++i)
		if (res[i] == minlevel) ans.push_back(i);
	return ans;*/
	//拓扑排序 内涵贪心算法
	if (edges.size() == 0) return{ 0 };
	vector<int> indegrees(n, 0);
	vector<vector<int>> adjacency(n);
	for (int i = 0; i < edges.size(); ++i) {
		indegrees[edges[i][0]]++;
		indegrees[edges[i][1]]++;
		adjacency[edges[i][0]].push_back(edges[i][1]);
		adjacency[edges[i][1]].push_back(edges[i][0]);
	}
	queue<int> q;
	for (int i = 0; i < n; ++i)
		if (indegrees[i] == 1) q.push(i);
	int cnt = q.size();
	while (n > 2)
	{
		n -= cnt;
		while (cnt--)
		{
			int temp = q.front();
			q.pop();
			indegrees[temp] = 0;
			for (int i = 0; i < adjacency[temp].size(); ++i) {
				if (indegrees[adjacency[temp][i]] != 0) {
					if (--indegrees[adjacency[temp][i]] == 1)
						q.push(adjacency[temp][i]);
				}
			}
		}
		cnt = q.size();
	}
	vector<int> ans;
	while (!q.empty()) {
		ans.push_back(q.front());
		q.pop();
	}
	return ans;
}

int CAlgorithmclass::maxCoins(vector<int>& nums)
{
	nums.insert(nums.begin(), 1);
	nums.push_back(1);
	vector<vector<int>> dp(nums.size(), vector<int>(nums.size(), 0));
	return back_maxCoins(nums, 0, nums.size() - 1, dp);
}

int CAlgorithmclass::nthSuperUglyNumber(int n, vector<int>& primes)
{
	//堆
	/*priority_queue<int,vector<int>,greater<int> > q;
	int cur = 1;
	while (--n)
	{
		for (auto& i:primes)
			q.push(cur*i);
		cur = q.top();
		q.pop();
		while (q.top() == cur)
			q.pop();
	}
	return cur;*/
	//动态规划
	int sSize = primes.size();
	vector<int> dp(n), index(sSize, 0);
	dp[0] = 1;
	for (int i = 1; i < n; ++i)
	{
		int imin = INT_MAX;
		for (int j = 0; j < sSize; ++j)
			imin = min(imin, dp[index[j]] * primes[j]);
		dp[i] = imin;
		for (int j = 0; j < sSize; ++j) {
			if (imin == dp[index[j]] * primes[j])
				index[j]++;
		}
	}
	return dp[n - 1];
}

vector<int> CAlgorithmclass::countSmaller(vector<int>& nums)
{
	/*if (nums.size() == 0) return{};
	vector<int> temp(nums.size());
	m_intVt = vector<int>(nums.size(),0);
	vector<int> index = vector<int>(nums.size(),0);
	for (int i = 0; i < nums.size(); i++)
		index[i] = i;
	back_countSmaller(nums, 0, nums.size() - 1, temp, index);
	vector<int> ans(nums.size());
	for (int i = 0; i < nums.size(); ++i) {
		ans[index[i]] = m_intVt[i];
	}
	return ans;*/
	//线段树
	if (nums.size() == 0) return{};
	vector<int> ans(nums.size());
	int imax = INT_MIN, imin = INT_MAX;
	for (auto& num : nums) {
		imax = max(num, imax);
		imin = min(num, imin);
	}
	SegmentTree* obj = new SegmentTree();
	SegmentTreeNode* root = obj->build(imin, imax);
	for (int i = nums.size() - 1; i >= 0; --i) {
		ans[i] = obj->count(root, imin, nums[i] - 1);
		obj->insert(root, nums[i], 1);
	}
	return ans;
}

string CAlgorithmclass::removeDuplicateLetters(string s)
{
	stack<char> st;
	unordered_map<char, int> amap;
	unordered_set<char> aset;
	for (int i = 0; i < s.size(); ++i)
		amap[s[i]] = i;
	for (int i = 0; i < s.size(); ++i) {
		if (aset.find(s[i]) == aset.end()) {
			while (!st.empty() && s[i]<st.top() && amap[st.top()]>i)
			{
				aset.erase(st.top());
				st.pop();
			}
			aset.insert(s[i]);
			st.push(s[i]);
		}
	}
	string ans = "";
	while (!st.empty())
	{
		ans = st.top() + ans;
		st.pop();
	}
	return ans;
}

int CAlgorithmclass::maxProduct1(vector<string>& words)
{
	//使用哈希函数 逐一比较
	/*if (words.size()<2) return 0;
	int fir = 0, sec = 0, ans = 0;
	for (int i = 0; i < words.size(); ++i)
	{
		fir = words[i].size();
		sec = 0;
		unordered_set<char> aset;
		for (auto& ch : words[i])
			aset.insert(ch);
		for (int j = i + 1; j < words.size(); ++j) {
			int temp = words[j].size();
			if (sec >= temp) continue;
			bool sign = false;
			for (int k = 0; k < words[j].size(); ++k) {
				if (aset.find(words[j][k]) != aset.end()) {
					sign = true;
					break;
				}
			}
			if (sign) continue;
			sec = temp;
		}
		ans = max(ans, fir*sec);
	}
	return ans;*/
	//使用位运算
	if (words.size() == 0)return 0;
	int n = words.size();
	vector<int> hash(n, 0);
	for (int i = 0; i < n; ++i) {
		for (auto& ch : words[i])
			hash[i] |= 1 << (ch - 'a');
	}
	int ans = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < n; ++j) {
			if (hash[i] & hash[j] == 0) {
				int temp = words[i].size()*words[j].size();
				ans = max(ans, temp);
			}
		}
	}
	return ans;
}

int CAlgorithmclass::bulbSwitch(int n)
{
	if (n == 0) return 0;
	return (int)sqrt(n);
}

void CAlgorithmclass::wiggleSort(vector<int>& nums)
{
	if (nums.size() < 2) return;
	int sSize = nums.size();
	int k = sSize / 2;
	//sort(nums.begin(), nums.end(), cmp);
	vector<int> tnums = nums;
	for (int i = 0; i < sSize; ++i) {
		if (i < k)
			nums[i * 2 + 1] = tnums[i];
		else
			nums[(i - k) * 2] = tnums[i];
	}
}

double CAlgorithmclass::myPow(double x, int n)
{
	if (x == 1.0 || x == 0.0) return 1;
	if (n < 0) {
		x = 1 / x;
		n = -n;
	}
	return help_mypow(x, n);
}

int CAlgorithmclass::countRangeSum(vector<int>& nums, int lower, int upper)
{
	int n = nums.size();
	vector<long long> S(n + 1, 0);
	vector<long long> assist(n + 1, 0);
	for (int i = 1; i <= n; i++)
		S[i] = S[i - 1] + nums[i - 1];
	return back_countRangeSum(S, assist, 0, n, lower, upper);
}

ListNode * CAlgorithmclass::oddEvenList(ListNode * head)
{
	ListNode* even = new ListNode(0);
	ListNode* odd = new ListNode(0);
	ListNode *even_t = even, *odd_t = odd;
	bool count = true;
	while (head)
	{
		if (count) {
			odd->next = head;
			odd = odd->next;
		}
		else
		{
			even->next = head;
			even = even->next;
		}
		count = !count;
		head = head->next;
	}
	even->next = nullptr;
	odd->next = even_t->next;
	ListNode* ans = odd_t->next;
	delete odd_t;
	delete even_t;
	return ans;
}

int CAlgorithmclass::longestIncreasingPath(vector<vector<int>>& matrix)
{
	if (matrix.size() == 0 || matrix[0].size() == 0) return 0;
	m_pairVt = { { 1,0 },{ -1,0 },{ 0,1 },{ 0,-1 } };
	m_int = 1;
	int row = matrix.size(), col = matrix[0].size();
	vector<vector<int>> sign(row, vector<int>(col, 1));
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			back_longestIncreasingPath(matrix, sign, i, j);
			m_int = max(m_int, sign[i][j]);
		}
	}
	return m_int;
}

int CAlgorithmclass::subarraySum(vector<int>& nums, int k)
{
	//暴力法
	/*int ans = 0;
	vector<long long> lnums(nums.size() + 1);
	lnums[0] = 0;
	for (int i = 0; i < nums.size(); ++i) {
		lnums[i+1] = lnums[i] + nums[i];
	}
	for (int i = 1; i < lnums.size(); ++i) {
		for (int j = i - 1; j >= 0; --j) {
			if (lnums[i] - lnums[j] == k)
				++ans;
		}
	}
	return ans;*/
	//哈希表
	unordered_map<int, int> amap;
	amap[0] = 1;
	int pre = 0, ans = 0;
	for (auto& n : nums) {
		pre += n;
		if (amap.find(pre - k) != amap.end())
			ans += amap[pre - k];
		amap[pre]++;
	}
	return ans;
}

ListNode * CAlgorithmclass::reverseKGroup(ListNode * head, int k)
{
	ListNode* cur = head;
	int count = 0;
	while (cur&&count < k)
	{
		cur = cur->next;
		count++;
	}
	if (count == k) {
		ListNode* res = reverseKGroup(cur, k);
		while (count--)
		{
			ListNode* temp = head->next;
			head->next = res;
			res = head;
			head = temp;
		}
		head = res;
	}
	return head;
}

bool CAlgorithmclass::isValidSerialization(string preorder)
{
	int slot = 1;
	for (int i = 0; i < preorder.size(); ++i) {
		if (preorder[i] == ',') {
			slot--;
			if (slot < 0) return false;
			if (preorder[i - 1] != '#')
				slot += 2;
		}
	}
	slot = preorder[preorder.size() - 1] == '#' ? slot - 1 : slot + 1;
	return slot == 0;
}

vector<string> CAlgorithmclass::findItinerary(vector<vector<string>>& tickets)
{
	int tsize = tickets.size();
	unordered_map<string, multiset<string>> adjaceney;
	for (int i = 0; i < tsize; ++i)
		adjaceney[tickets[i][0]].insert(tickets[i][1]);
	vector<string> ans;
	stack<string> s;
	s.push("JFK");
	while (!s.empty())
	{
		string str = s.top();
		if (adjaceney.find(str) != adjaceney.end() && !adjaceney[str].empty()) {
			string temp = *adjaceney[str].begin();
			adjaceney[str].erase(adjaceney[str].begin());
			s.push(temp);
		}
		else {
			ans.push_back(str);
			s.pop();
		}
	}
	return vector<string>(ans.rbegin(), ans.rend());
}

bool CAlgorithmclass::increasingTriplet(vector<int>& nums)
{
	if (nums.size() < 3) return false;
	int small = INT_MAX, mid = INT_MAX;
	for (int i = 0; i < nums.size(); ++i) {
		if (nums[i] <= small) {
			small = nums[i];
		}
		else if (nums[i] <= mid)
			mid = nums[i];
		else
			return true;
	}
	return false;
}

bool CAlgorithmclass::validPalindrome(string s)
{
	int l = 0, r = s.size() - 1;
	bool once = true;
	return back_validPalindrome(s, l, r, once);
}

int CAlgorithmclass::findTheLongestSubstring(string s)
{
	int ans = 0, status = 0;
	vector<int> pos(1 << 5, -1);
	pos[0] = 0;
	for (int i = 0; i < s.size(); ++i) {
		if (s[i] == 'a')
			status ^= 1 << 0;
		else if (s[i] == 'e')
			status ^= 1 << 1;
		else if (s[i] == 'i')
			status ^= 1 << 2;
		else if (s[i] == 'o')
			status ^= 1 << 3;
		else if (s[i] == 'u')
			status ^= 1 << 4;
		if (pos[status] != -1)
			ans = max(ans, i + 1 - pos[status]);
		else
			pos[status] = i + 1;
	}
	return ans;
}

vector<int> CAlgorithmclass::countBits(int num)
{
	vector<int> ans(num + 1, 0);
	for (int i = 0; i <= num; ++i) {
		ans[i] = ans[i&(i - 1)] + 1;
	}
	return ans;
}

double CAlgorithmclass::findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2)
{
	//时间复杂度为O(m+n)
	/*int m = nums1.size(), n = nums2.size();
	vector<int> res(m + n);
	int i = 0, j = 0,pos = 0;
	while (i<m&&j<n)
	{
		if (nums1[i] < nums2[j])
			res[pos] = nums1[i++];
		else
			res[pos] = nums2[j++];
		pos++;
	}
	for (; i < m; ++i)
		res[pos++] = nums1[i];
	for (; j < n; ++j)
		res[pos++] = nums2[j];
	int temp = (m + n) / 2;
	return (m + n) % 2 ? (double)res[temp] : (double)(res[temp] + res[temp - 1]) / 2;*/
	//时间复杂度为O(log(m+n))
	if (nums1.size() > nums2.size())
		return findMedianSortedArrays(nums2, nums1);
	int len = nums1.size() + nums2.size();
	int cutL = 0, cutR = nums1.size();
	int cut1mid = 0, cut2mid = 0;
	while (cut1mid <= nums1.size())
	{
		cut1mid = cutL + (cutR - cutL) / 2;
		cut2mid = len / 2 - cut1mid;
		double l1 = cut1mid == 0 ? INT_MIN : nums1[cut1mid - 1];
		double r1 = cut1mid == nums1.size() ? INT_MAX : nums1[cut1mid];
		double l2 = cut2mid == 0 ? INT_MIN : nums2[cut2mid - 1];
		double r2 = cut2mid == nums2.size() ? INT_MAX : nums2[cut2mid];
		if (l1 > r2)
			cutR = cut1mid - 1;
		else if (l2 > r1)
			cutL = cut1mid + 1;
		else {
			if (len % 2 == 0)
				return (max(l1, l2) + min(r1, r2)) / 2.0;
			else
				return min(r1, r2);
		}
	}
	return 0.0;
}

vector<int> CAlgorithmclass::topKFrequent(vector<int>& nums, int k)
{
	/*struct comp
	{
		bool operator() (unordered_map<int, int>::iterator it1, unordered_map<int, int>::iterator it2) {
			return it1->second < it2->second;
		}
	};

	vector<int> ans(k);
	unordered_map<int, int> amap;
	for (auto &n : nums)
		amap[n]++;
	priority_queue<unordered_map<int, int>::iterator, vector<unordered_map<int, int>::iterator>, comp> q;
	for (unordered_map<int, int>::iterator it = amap.begin(); it != amap.end(); ++it)
		q.push(it);
	for (int i = 0; i < k; ++i) {
		ans[i] = q.top()->first;
		q.pop();
	}
	return ans;*/
	vector<int> ret;
	unordered_map<int, int> mp;
	priority_queue<pair<int, int>> pq;
	for (auto i : nums) mp[i]++;
	for (auto p : mp) {
		pq.push(pair<int, int>(-p.second, p.first));
		if (pq.size() > k) pq.pop();
	}
	while (k--) {
		ret.push_back(pq.top().second);
		pq.pop();
	}
	return ret;
}

string CAlgorithmclass::decodeString(string s)
{
	int i(0);
	while (i < s.size())
	{
		if (s[i] == ']') {
			int j = i;
			while (s[j] != '[')
			{
				j--;
			}
			string s1 = s.substr(j + 1, i - j - 1);
			int k = j - 1;
			while (k >= 0 && isdigit(s[k]))
			{
				k--;
			}
			int chong = stoi(s.substr(k + 1, j - k - 1));
			string temp = "";
			while (chong--)
				temp += s1;
			s = s.replace(k + 1, i - k, temp);
			i = k;
		}
		i++;
	}
	return s;
}

int CAlgorithmclass::subarraysDivByK(vector<int>& A, int K)
{
	unordered_map<int, int> amap;
	amap[0] = 1;
	int count = 0, sum = 0;
	for (auto& n : A) {
		sum += n;
		int modnum = (sum%K + K) % K;
		if (amap.find(modnum) != amap.end())
			count += amap[modnum];
		amap[modnum]++;
	}
	return count;
}

bool CAlgorithmclass::canPartition(vector<int>& nums)
{
	/*int sum = 0;
	for (auto& n : nums) {
		sum += n;
	}
	if (sum % 2) return false;
	int half = sum / 2;
	vector<vector<bool>> dp(nums.size(), vector<bool>(half + 1, false));
	if (nums[0] <= half)
		dp[0][nums[0]] = true;
	for (int i = 1; i < nums.size(); ++i) {
		for (int j = 0; j <= half; ++j) {
			if (nums[i] == j) {
				dp[i][j] = true;
				continue;
			}
			if (nums[i] < j)
				dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i]];
		}
	}
	return dp[nums.size() - 1][half];*/
	unordered_map<int, int> vis1;
	unordered_map<int, int> vis2;
	int len = nums.size();
	int sum = 0, ans = 0;
	for (int i = 0; i < len; i++)
		sum += nums[i];
	if (sum % 2 == 1) return false;
	return back_canPartition(sum / 2, vis1, vis2, nums);
}

int CAlgorithmclass::pathSum3(TreeNode * root, int sum)
{
	vector<int> temp;
	m_int = 0;
	back_pathSum3(root, sum, 1, temp);
	return m_int;
}

vector<int> CAlgorithmclass::findAnagrams(string s, string p)
{
	/*unordered_map<char, int> amap,bmap;
	for (auto&c : p)
		amap[c]++;
	vector<int> ans;
	int psize = p.size();
	for (int i = 0; i < s.size(); ++i) {
		int j = i - psize;
		if (j >= 0 && amap.find(s[j]) != amap.end())
			bmap[s[j]]--;
		if (amap.find(s[i]) != amap.end())
			bmap[s[i]]++;
		if (amap == bmap)
			ans.push_back(i - psize + 1);
	}
	return ans;*/
	vector<int> ans;
	int len_s = s.length(), len_p = p.length();
	if (len_s < len_p) return ans;
	vector<int> freq_s(26, 0), freq_p(26, 0);
	for (int i = 0; i < len_p; i++) {
		freq_p[p[i] - 'a']++;
		freq_s[s[i] - 'a']++;
	}
	int dif = 0;
	for (int i = 0; i < 26; i++) {
		if (freq_s[i] != freq_p[i])
			dif++;
	}
	if (dif == 0) ans.push_back(0);
	for (int l = 0, r = len_p; r < len_s; r++, l++) {
		freq_s[s[l] - 'a']--;
		freq_s[s[r] - 'a']++;
		if (s[l] != s[r]) {
			if (freq_s[s[l] - 'a'] == freq_p[s[l] - 'a']) dif--;
			else if (freq_s[s[l] - 'a'] + 1 == freq_p[s[l] - 'a']) dif++;
			if (freq_s[s[r] - 'a'] == freq_p[s[r] - 'a']) dif--;
			else if (freq_s[s[r] - 'a'] - 1 == freq_p[s[r] - 'a']) dif++;
		}
		if (dif == 0) ans.push_back(l + 1);
	}
	return ans;
}

vector<int> CAlgorithmclass::findDisappearedNumbers(vector<int>& nums)
{
	/*int n = nums.size();
	vector<bool> temp(n + 1, false);
	for (int i = 0; i < n; ++i) {
		temp[nums[i]] = true;
	}
	vector<int> ans;
	for (int i = 1; i <= n; ++i) {
		if (!temp[i])
			ans.push_back(i);
	}
	return ans;*/
	for (int i = 0; i < nums.size(); ++i) {
		if (nums[i] - 1 == i)
			continue;
		while (nums[i] != -1 && nums[i] != nums[nums[i] - 1])
		{
			swap(nums[i], nums[nums[i] - 1]);
		}
		if (nums[i] - 1 != i)
			nums[i] = -1;
	}
	vector<int> ans;
	for (int i = 0; i < nums.size(); ++i) {
		if (nums[i] == -1)
			ans.push_back(i + 1);
	}
	return ans;
}

int CAlgorithmclass::hammingDistance(int x, int y)
{
	int temp = x ^ y;
	int ans = 0;
	while (temp) {
		temp &= temp - 1;
		ans++;
	}
	return ans;
}

int CAlgorithmclass::findTargetSumWays(vector<int>& nums, int S)
{
	/*m_int = 0;
	back_findTargetSumWays(nums, 0, S, 0);
	return m_int;*/
	/*vector<vector<int>> dp(nums.size(), vector<int>(2001, 0));
	dp[0][nums[0] + 1000] = 1;
	dp[0][-nums[0] + 1000] += 1;
	for (int i = 1; i < nums.size(); ++i) {
		for (int j = -1000; j <= 1000; ++j) {
			if (dp[i - 1][j + 1000] > 0) {
				dp[i][j + nums[i] + 1000] += dp[i - 1][j + 1000];
				dp[i][j - nums[i] + 1000] += dp[i - 1][j + 1000];
			}
		}
	}
	return S > 1000 ? 0 : dp[nums.size() - 1][S + 1000];*/
	vector<int> dp(2001, 0);
	dp[nums[0] + 1000] = 1;
	dp[-nums[0] + 1000] += 1;
	for (int i = 1; i < nums.size(); ++i) {
		vector<int> temp = vector<int>(2001, 0);
		for (int j = -1000; j <= 1000; ++j) {
			if (dp[j + 1000] > 0) {
				temp[j + nums[i] + 1000] += dp[j + 1000];
				temp[j - nums[i] + 1000] += dp[j + 1000];
			}
		}
		dp = temp;
	}
	return S > 1000 ? 0 : dp[S + 1000];
}

int CAlgorithmclass::findUnsortedSubarray(vector<int>& nums)
{
	int ans = 0;
	int r = 0, l = nums.size() - 1;
	int imax = INT_MIN;
	for (int i = 0; i < nums.size(); ++i) {
		if (nums[i] < imax)
			r = i;
		imax = max(imax, nums[i]);
	}
	int imin = INT_MAX;
	for (int i = nums.size() - 1; i >= 0; --i) {
		if (nums[i] > imin)
			l = i;
		imin = min(imin, nums[i]);
	}
	return r > l ? r - l + 1 : 0;
}

TreeNode * CAlgorithmclass::mergeTrees(TreeNode * t1, TreeNode * t2)
{
	if (!t1)
		return t2;
	if (!t2)
		return t1;
	t1->val += t2->val;
	t1->left = mergeTrees(t1->left, t2->left);
	t1->right = mergeTrees(t1->right, t2->right);
	return t1;
}

int CAlgorithmclass::leastInterval(vector<char>& tasks, int n)
{
	int s[26] = { 0 };
	int imax = 0;
	for (auto& c : tasks) {
		int temp = c - 'A';
		s[temp]++;
		if (s[temp] > imax) imax = s[temp];
	}
	int ans = (imax - 1)*(n + 1);
	for (int i = 0; i < 26; ++i)
		if (s[i] == imax) ans++;
	return ans > tasks.size() ? ans : tasks.size();
}

int CAlgorithmclass::translateNum(int num)
{
	if (num < 10) return 1;
	string s = to_string(num);
	vector<int> dp(s.size(), 1);
	dp[1] = stoi(s.substr(0, 2)) <= 25 ? 2 : 1;
	for (int i = 2; i < s.size(); ++i) {
		dp[i] = dp[i - 1];
		if (s[i - 1] != '0'&&stoi(s.substr(i - 1, 2)) <= 25)
			dp[i] += dp[i - 2];
	}
	return dp[s.size() - 1];
}

vector<int> CAlgorithmclass::dailyTemperatures(vector<int>& T)
{
	/*int len = T.size() + 1;
	vector<int> ans(T.size(), len);
	vector<int> temp(101, 0);
	for (int i = T.size() - 1; i >= 0; --i) {
		temp[T[i]] = i;
		for (int j = T[i] + 1; j <= 100; ++j) {
			if (temp[j] != 0) {
				ans[i] = min(temp[j] - i, ans[i]);
			}
		}
		if (ans[i] == len) ans[i] = 0;
	}*/
	//递减栈
	vector<int> ans(T.size(), 0);
	stack<int> s;
	for (int i = 0; i < T.size(); ++i) {
		while (!s.empty() && T[s.top()] < T[i]) {
			int temp = s.top();
			s.pop();
			ans[temp] = i - temp;
		}
		s.push(i);
	}
	return ans;
}

ListNode * CAlgorithmclass::removeDuplicateNodes(ListNode * head)
{
	if (head == nullptr) return head;
	ListNode* ans = head;
	//unordered_set<int> aset;
	//aset.insert(head->val);
	vector<bool> aset(20001, false);
	aset[head->val] = true;
	while (head != nullptr&&head->next != nullptr) {
		if (/*aset.find(head->next->val) != aset.end()*/aset[head->next->val]) {
			ListNode* node = head->next;
			head->next = node->next;
			delete node;
		}
		else {
			//aset.insert(head->next->val);
			aset[head->next->val] = true;
			head = head->next;
		}
	}
	return ans;
}

bool CAlgorithmclass::robot(string command, vector<vector<int>>& obstacles, int x, int y)
{
	/*vector<vector<bool>> map(x + 1, vector<bool>(y + 1, true));
	for (int i = 0; i<obstacles.size(); ++i) {
		if (obstacles[i][0] > x || obstacles[i][1] > y) continue;
		map[obstacles[i][0]][obstacles[i][1]] = false;
	}
	int x0 = 0, y0 = 0;
	int sSize = command.size() - 1;
	for (int i = 0; i<command.size(); ++i) {
		if (command[i] == 'U') {
			y0++;
			if (y0>y) return false;
		}
		else {
			x0++;
			if (x0>x) return false;
		}
		if (!map[x0][y0]) return false;
		if (x0 == x&&y0 == y) return true;
		if (i == sSize) i = -1;
	}
	return 0;*/
	unordered_set<long> s;
	int xx = 0, yy = 0;
	s.insert(0);
	for (auto c : command) {
		if (c == 'U') yy++;
		else if (c == 'R')xx++;
		s.insert(((long)xx << 30) | yy);
	}

	int circle = min(x / xx, y / yy);
	if (s.count(((long)(x - circle * xx) << 30) | (y - circle * yy)) == 0) return false;

	for (auto v : obstacles) {
		if (v.size() != 2) continue;
		if (v[0] > x || v[1] > y) continue;
		circle = min(v[0] / xx, v[1] / yy);
		if (s.count(((long)(v[0] - circle * xx) << 30) | (v[1] - circle * yy))) return false;
	}
	return true;
}

bool CAlgorithmclass::isMatch(string s, string p)
{
	int m = s.size(), n = p.size();
	vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
	dp[0][0] = true;
	for (int i = 1; i <= n; ++i) {
		if (p[i - 1] == '*')
			dp[0][i] = true;
		else break;
	}
	for (int i = 1; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			if (p[j - 1] == '?' || p[j - 1] == s[i - 1]) {
				dp[i][j] = dp[i - 1][j - 1];
			}
			else if (p[j - 1] == '*') {
				dp[i][j] = dp[i][j - 1] | dp[i - 1][j];
			}
		}
	}
	return dp[m][n];
}

bool CAlgorithmclass::isMatch2(string s, string p)
{
	s = " " + s;
	p = " " + p;
	int m = s.size();
	int n = p.size();
	vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
	dp[0][0] = true;
	for (int i = 0; i < n; ++i) {
		if (p[i] == '*') dp[0][i] = true;
		else break;
	}
	for (int i = 1; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			if (p[j - 1] == s[i - 1] || p[j - 1] == '.')
				dp[i][j] = dp[i - 1][j - 1];
			else if (p[j - 1] == '*') {
				if (p[j - 2] == '.' || p[j - 2] == s[i - 1])
					dp[i][j] = dp[i][j - 1] | dp[i - 1][j] | dp[i][j - 2];
				else dp[i][j] = dp[i][j - 2];
			}
		}
	}
	return dp[m][n];
}

int CAlgorithmclass::findBestValue(vector<int>& arr, int target)
{
	int len = arr.size();
	QuickSort(arr, 0, len - 1);
	int sum = 0;
	for (int i = 0; i < len; ++i) {
		int x = (target - sum) / (len - i);
		if (x < arr[i]) {
			double tmp_x = (double)(target - sum) / (double)(len - i);
			if (abs((double)x - tmp_x) > 0.5)
				return x + 1;
			else
				return x;
		}
		sum += arr[i];
	}
	return arr[len - 1];
}

int CAlgorithmclass::maxCoins2(vector<int>& nums)
{
	int n = nums.size();
	m_intVt.resize(n + 2);
	for (int i = 0; i < n; ++i)
		m_intVt[i + 1] = nums[i];
	m_intVt[0] = m_intVt[n + 1] = 1;
	m_intVtVt.resize(n + 2, vector<int>(n + 2, -1));
	return solve_maxCoins2(0, n + 1);
}

int CAlgorithmclass::splitArray(vector<int>& nums, int m)
{
	int n = nums.size();
	vector<vector<long long>> dp(n + 1, vector<long long>(m + 1, LLONG_MAX));
	vector<long long> sub(n + 1, 0);
	for (int i = 0; i < n; ++i) {
		sub[i + 1] = sub[i] + nums[i];
	}
	dp[0][0] = 0;
	for (int i = 1; i <= n; ++i) {
		for (int j = 1; j <= min(i, m); ++j) {
			for (int k = j - 1; k < i; ++k) {
				dp[i][j] = min(dp[i][j], max(dp[k][j - 1], sub[i] - sub[k]));
			}
		}
	}
	return (int)dp[n][m];
}

vector<int> CAlgorithmclass::smallestRange(vector<vector<int>>& nums)
{
	int l = 0, r = INT_MAX;
	int sSize = nums.size();
	vector<int> sub(sSize,0);
	auto cmp = [&](const int& u, const int&v) {
		return nums[u][sub[u]] > nums[v][sub[v]];
	};
	priority_queue<int, vector<int>, decltype(cmp)> q(cmp);//小顶堆
	int imin = 0, imax = INT_MIN;
	for (int i = 0; i < sSize; ++i) {
		q.push(i);
		imax = max(imax, nums[i][0]);
	}
	while (true)
	{
		int k = q.top();
		q.pop();
		imin = nums[k][sub[k]];
		if (imax - imin < r - l) {
			r = imax;
			l = imin;
		}
		if (sub[k] == nums[k].size() - 1)
			break;
		++sub[k];
		imax = max(imax, nums[k][sub[k]]);
		q.push(k);
	}
	return {l,r};
}

string CAlgorithmclass::addStrings(string num1, string num2)
{
	/*if (num1.size() > num2.size())
		return addStrings(num2, num1);
	reverse(num1.begin(), num1.end());
	reverse(num2.begin(), num2.end());
	int cnt = 0;
	string ans = "";
	for (int i = 0; i < num1.size(); ++i) {
		int res = num1[i] - '0' + num2[i] - '0' + cnt;
		if (res > 9)
			cnt = res / 10;
		else
			cnt = 0;
		res %= 10;
		ans = to_string(res) + ans;
	}
	if (num2.size() == num1.size()) {
		if(cnt)
			ans = to_string(cnt) + ans;
	}
	else {
		for (int i = num1.size(); i < num2.size(); ++i) {
			int tmp = num2[i] - '0' + cnt;
			if (tmp > 9) cnt = tmp / 10;
			else cnt = 0;
			tmp %= 10;
			ans = to_string(tmp) + ans;
		}
		if (cnt) ans = to_string(cnt) + ans;
	}
	return ans;*/
	//
	int i = num1.length() - 1, j = num2.length() - 1, add = 0;
	string ans = "";
	while (i >= 0 || j >= 0 || add != 0) {
		int x = i >= 0 ? num1[i] - '0' : 0;
		int y = j >= 0 ? num2[j] - '0' : 0;
		int result = x + y + add;
		ans.push_back('0' + result % 10);
		add = result / 10;
		i -= 1;
		j -= 1;
	}
	// 计算完以后的答案需要翻转过来
	reverse(ans.begin(), ans.end());
	return ans;
}

vector<vector<int>> CAlgorithmclass::palindromePairs(vector<string>& words)
{
	vector<vector<int>> ans;
	for (int i = 0;i<words.size();++i)
	{
		for (int j = 0; j < words.size(); ++j)
		{
			if (i == j) continue;
			if (isPalindrome(words[i], words[j]))
				ans.push_back({ i,j });
		}
	}
	return ans;
}

int CAlgorithmclass::longestPalindromeSubseq(string s)
{
	int len = s.size();
	vector<vector<int>> dp(len, vector<int>(len,0));
	for (int i = 0; i < len; ++i)
		dp[i][i] = 1;
	for (int i = len - 1; i >= 0; --i) {
		for (int j = i + 1; j < len; ++j) {
			if (s[i] == s[j])
				dp[i][j] = dp[i + 1][j - 1] + 2;
			else
				dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
		}
	}
	return dp[0][len - 1];
}









bool CAlgorithmclass::isPalindrome(string & a, string & b)
{
	string sum = a + b;
	int sSize = sum.size();
	for (int i = 0; i < sSize / 2; ++i)
	{
		if (sum[i] != sum[sSize - i - 1])
			return false;
	}
	return true;
}

int CAlgorithmclass::solve_maxCoins2(int l, int r)
{
	if (l >= r - 1) return 0;
	if (m_intVtVt[l][r] != -1) return m_intVtVt[l][r];
	int sum = 0;
	for (int i = l + 1; i < r; ++i) {
		sum = max(sum, m_intVt[i] * m_intVt[l] * m_intVt[r] + solve_maxCoins2(l, i) + solve_maxCoins2(i, r));
	}
	m_intVtVt[l][r] = sum;
	return sum;
}

void CAlgorithmclass::back_findTargetSumWays(vector<int>& nums, int a, int S, int sum)
{
	if (a >= nums.size()) {
		if (sum == S)
			m_int++;
		return;
	}
	back_findTargetSumWays(nums, a + 1, S, sum + nums[a]);
	back_findTargetSumWays(nums, a + 1, S, sum - nums[a]);
}

void CAlgorithmclass::back_pathSum3(TreeNode * root, int sum, int level, vector<int>& temp)
{
	if (root == nullptr)
		return;
	if (level >= temp.size())
		temp.push_back(0);
	for (int i = 0; i < level; ++i) {
		temp[i] += root->val;
		if (sum - temp[i] == 0)
			m_int++;
	}
	back_pathSum3(root->left, sum, level + 1, temp);
	back_pathSum3(root->right, sum, level + 1, temp);
	for (int i = 0; i < level; ++i)
		temp[i] -= root->val;
}

bool CAlgorithmclass::back_canPartition(int num, unordered_map<int, int>&vis1, unordered_map<int, int>&vis2, vector<int>& nums)
{
	if (num == 0) return true;
	for (int i = 0; i < nums.size(); i++) {
		int m = num - nums[i];
		if (m < 0 || vis1[i] == 1 || vis2[m] == 1)
			continue;
		else if (m == 0)
			return true;
		vis1[i] = 1;
		vis2[m] = 1;
		if (back_canPartition(m, vis1, vis2, nums) == true)
			return true;
		vis1[i] = 0;
	}
	return false;
}

string CAlgorithmclass::back_decode(string s, int a)
{
	if (s[a] == ']')
		return "";
	string res;
	int k = 0;
	for (int i = a; i < s.size(); ++i) {

		if (isdigit(s[i])) {
			int start = i;
			while (i < s.size() && isdigit(s[i]))
				i++;
			k = stoi(s.substr(start, i - start));
			string temp = back_decode(s, i);
			while (k--)
			{
				res += temp;
			}
		}
		if (s[i] == '[') {
			int l = i + 1;
			while (s[i] != ']')
				i++;
			string str = s.substr(l, i - l);
		}

	}
	return string();
}

vector<int> CAlgorithmclass::back_rob3(TreeNode * root)
{
	if (root == nullptr)
		return { 0,0 };
	vector<int> res(2);
	vector<int> l = back_rob3(root->left);
	vector<int> r = back_rob3(root->right);
	res[0] = max(l[0] + r[0], root->val + l[1] + r[1]);
	res[1] = l[0] + r[0];
	return res;
}

bool CAlgorithmclass::back_validPalindrome(string & s, int l, int r, bool once)
{
	if (l >= r) {
		return true;
	}
	if (s[l] == s[r]) {
		if (back_validPalindrome(s, l + 1, r - 1, once))
			return true;
	}
	if (once&&l <= r - 1 && s[l] == s[r - 1]) {
		if (back_validPalindrome(s, l + 1, r - 2, once = false))
			return true;
	}
	if (once&&l + 1 <= r && s[l + 1] == s[r]) {
		if (back_validPalindrome(s, l + 2, r - 1, once = false))
			return true;
	}
	return false;
}

void CAlgorithmclass::back_longestIncreasingPath(vector<vector<int>>& matrix, vector<vector<int>>& sign, int i, int j)
{
	if (sign[i][j] > 1)
		return;
	int t = 0;
	for (int k = 0; k < 4; ++k) {
		int ti = m_pairVt[k].first + i;
		int tj = m_pairVt[k].second + j;
		if (ti < 0 || tj < 0 || ti >= matrix.size() || tj >= matrix[0].size() || !sign[ti][tj] || matrix[ti][tj] <= matrix[i][j])
			continue;
		back_longestIncreasingPath(matrix, sign, ti, tj);
		t = max(t, sign[ti][tj]);
	}
	sign[i][j] += t;
}

int CAlgorithmclass::back_countRangeSum(vector<long long> &S, vector<long long> &assist, int L, int R, int low, int up)
{
	if (L >= R) return 0;

	int cnt = 0;
	int M = L + (R - L) / 2;
	cnt += back_countRangeSum(S, assist, L, M, low, up);
	cnt += back_countRangeSum(S, assist, M + 1, R, low, up);
	int Left = L;
	int Upper = M + 1, Lower = M + 1;
	while (Left <= M) {
		while (Lower <= R && S[Lower] - S[Left] < low)Lower++;
		while (Upper <= R && S[Upper] - S[Left] <= up)Upper++;

		cnt += Upper - Lower;
		Left++;
	}
	//以下为归并排序中归并过程
	Left = L;
	int Right = M + 1;
	int pos = L;
	while (Left <= M || Right <= R) {
		if (Left > M)assist[pos] = S[Right++];
		if (Right > R && Left <= M)assist[pos] = S[Left++];

		if (Left <= M && Right <= R) {
			if (S[Left] <= S[Right])assist[pos] = S[Left++];
			else assist[pos] = S[Right++];
		}
		pos++;
	}
	for (int i = L; i <= R; i++)S[i] = assist[i];
	return cnt;
}

double CAlgorithmclass::help_mypow(double x, int n)
{
	if (n == 0)
		return 1.0;
	double temp = help_mypow(x, n / 2);
	return n % 2 ? temp * temp*x : temp * temp;
}

void CAlgorithmclass::back_countSmaller(vector<int>& nums, int l, int r, vector<int>& temp, vector<int>& index)
{
	if (l == r)
		return;
	int mid = l + (r - l) / 2;
	back_countSmaller(nums, l, mid, temp, index);
	back_countSmaller(nums, mid + 1, r, temp, index);
	if (nums[index[mid]] > nums[index[mid + 1]])
		return;
	vector<int> cp = m_intVt;
	int i = l, j = mid + 1;
	int pos = l;
	while (i <= mid && j <= r)
	{
		if (nums[index[i]] > nums[index[j]]) {
			temp[pos] = index[i];
			cp[pos] = m_intVt[i] + r - j + 1;
			i++;
		}
		else {
			temp[pos] = index[j];
			cp[pos] = m_intVt[j];
			j++;
		}
		pos++;
	}
	for (int k = i; k <= mid; ++k) {
		temp[pos] = index[k];
		cp[pos] = m_intVt[k];
		pos++;
	}
	for (int k = j; k <= r; ++k)
	{
		temp[pos] = index[k];
		cp[pos] = m_intVt[k];
		pos++;
	}
	m_intVt = cp;
	copy(temp.begin() + l, temp.begin() + r + 1, index.begin() + l);
	return;
}

int CAlgorithmclass::back_maxCoins(vector<int>& nums, int l, int r, vector<vector<int>>& dp)
{
	if (r - l == 1)return 0;
	if (dp[l][r] != 0) return dp[l][r];
	int imax = 0;
	for (int i = l + 1; i < r; ++i)
	{
		imax = max(nums[i] * nums[l] * nums[r] + back_maxCoins(nums, l, i, dp) + back_maxCoins(nums, i, r, dp), imax);
	}
	dp[l][r] = imax;
	return imax;
}

bool CAlgorithmclass::back_isSubtree(TreeNode * s, TreeNode * t)
{
	if (s == nullptr&&t == nullptr)
		return true;
	if (s != nullptr&&t != nullptr&& s->val != t->val)
		return false;
	if (s == nullptr || t == nullptr)
		return false;
	if (back_isSubtree(s->left, t->left) && back_isSubtree(s->right, t->right))
		return true;
	return false;
}

bool CAlgorithmclass::back_isAdditiveNumber(string num, vector<long double> tmp)
{
	int n = tmp.size();
	if (n >= 3 && tmp[n - 1] != tmp[n - 2] + tmp[n - 3]) return false;
	if (num.size() == 0 && n >= 3) return true;
	for (int i = 0; i < num.size(); ++i) {
		string cur = num.substr(0, i + 1);
		if (cur[0] == '0' && cur.size() != 1) continue;
		tmp.push_back(stold(cur));
		if (back_isAdditiveNumber(num.substr(i + 1), tmp)) return true;
		tmp.pop_back();
	}
	return false;
}

bool CAlgorithmclass::orEmpty(unordered_map<char, int>& amap)
{
	for (auto & i : amap)
	{
		if (i.second > 0)
			return false;
	}
	return true;
}

void CAlgorithmclass::backCombine(int n, int k, vector<int> temp, int a)
{
	if (k == temp.size())
	{
		m_intVtVt.push_back(temp);
		return;
	}
	for (int i = a; i < n - (k - temp.size()) + 1; i++)
	{
		temp.push_back(i);
		backCombine(n, k, temp, i + 1);
		temp.pop_back();
	}
}

void CAlgorithmclass::backSubsets(vector<int>& nums, vector<int>& temp, int a)
{
	m_intVtVt.push_back(temp);
	if (a == nums.size())
	{
		return;
	}
	for (int i = a; i < nums.size(); ++i)
	{
		temp.push_back(nums[i]);
		backSubsets(nums, temp, i + 1);
		temp.pop_back();
	}
}

//超出内存限制 //单词搜索
void CAlgorithmclass::backExist(vector<vector<char>>& board, vector<vector<int>>& _or, string word, int i, int j, int n, int m, int k)
{
	if (k == word.size())
	{
		m_bool = true;
		return;
	}
	if (i > 0 && board[i - 1][j] == word[k] && _or[i - 1][j] == 0)
	{
		_or[i - 1][j] = 1;
		backExist(board, _or, word, i - 1, j, n, m, k + 1);
		_or[i - 1][j] = 0;
	}
	if (j > 0 && board[i][j - 1] == word[k] && _or[i][j - 1] == 0)
	{
		_or[i][j - 1] = 1;
		backExist(board, _or, word, i, j - 1, n, m, k + 1);
	}
	if (i < n - 1 && board[i + 1][j] == word[k] && _or[i + 1][j] == 0)
	{
		_or[i + 1][j] = 1;
		backExist(board, _or, word, i + 1, j, n, m, k + 1);
	}
	if (j < m - 1 && board[i][j + 1] == word[k] && _or[i][j + 1] == 0)
	{
		_or[i][j + 1] = 1;
		backExist(board, _or, word, i, j + 1, n, m, k + 1);
	}
	else
		return;
}
//单词搜索
bool CAlgorithmclass::dfs(vector<vector<char>>& board, string& word, int size, int x, int y, vector<vector<int>>& f)
{
	if (size == word.size()) {
		return true;
	}//outofbound
	if (x < 0 || x >= board.size() || y < 0 || y >= board[0].size() || board[x][y] != word[size]) {
		return false;
	}
	if (f[x][y] == 0) {
		f[x][y] = 1;
		if (dfs(board, word, size + 1, x + 1, y, f)
			|| dfs(board, word, size + 1, x - 1, y, f)
			|| dfs(board, word, size + 1, x, y + 1, f)
			|| dfs(board, word, size + 1, x, y - 1, f)) {
			return true;
		}
		f[x][y] = 0;
	}
	return false;
}

void CAlgorithmclass::back_dfs(vector<vector<char>>& board, int i, int j, Trie * node)
{
	char c = board[i][j];
	if (c == '#' || node->next[c - 'a'] == nullptr)
		return;
	node = node->next[c - 'a'];
	if (node->str != "") {
		m_strVt.push_back(node->str);
		node->str = "";
	}
	board[i][j] = '#';
	if (i > 0) back_dfs(board, i - 1, j, node);
	if (j > 0) back_dfs(board, i, j - 1, node);
	if (i < m_row - 1) back_dfs(board, i + 1, j, node);
	if (j < m_col - 1) back_dfs(board, i, j + 1, node);
	board[i][j] = c;
}

void CAlgorithmclass::DFS_shudu(int i, int j, vector<vector<char>>& board)
{
	if (solved == true)
		return;
	if (i >= 9) {
		solved = true;
		return;
	}
	//board[i][j]非空，考虑下一个位置
	if (board[i][j] != '.') {
		if (j < 8)
			DFS_shudu(i, j + 1, board);
		else if (j == 8)
			DFS_shudu(i + 1, 0, board);
		if (solved == true)
			return;
	}
	//board[i][j]为空，可以填数
	else {
		int index = 3 * (i / 3) + j / 3;
		for (int num = 1; num <= 9; num++) {
			if (!row[i][num] && !col[j][num] && !box[index][num]) //num是否符合规则
			{
				board[i][j] = num + '0'; //填数
				row[i][num] = col[j][num] = box[index][num] = true;

				if (j < 8)   //递归
					DFS_shudu(i, j + 1, board);
				else if (j == 8)
					DFS_shudu(i + 1, 0, board);

				if (!solved) {     //回溯
					row[i][num] = col[j][num] = box[index][num] = false;
					board[i][j] = '.';
				}

			}
		}
	}
}

void CAlgorithmclass::backsubsetsWithDup(vector<int>& nums, vector<int>& temp, int a)
{
	m_intVtVt.push_back(temp);
	if (temp.size() > nums.size())
		return;
	for (int i = a; i < nums.size(); i++)
	{
		if (i > a && nums[i] == nums[i - 1])
			continue;
		temp.push_back(nums[i]);
		backsubsetsWithDup(nums, temp, i + 1);
		temp.pop_back();
	}
}

void CAlgorithmclass::back_IpAddresses(string s, int n, string segment)
{
	if (n == 4 && s.empty())
	{
		m_strVt.push_back(segment);
		return;
	}
	for (int i = 1; i < 4; i++)
	{
		if (s.size() < i)
			break;
		int temp = stoi(s.substr(0, i));
		if (temp > 255 || i != to_string(temp).size())
			continue;
		back_IpAddresses(s.substr(i), n + 1, segment + s.substr(0, i) + (n == 3 ? "" : "."));
	}
}

void CAlgorithmclass::middle_order(TreeNode * root)
{
	if (root == NULL)
		return;
	middle_order(root->left);
	m_intVt.push_back(root->val);
	middle_order(root->right);
}

vector<TreeNode*> CAlgorithmclass::back_generateTrees(int start, int end)
{
	vector<TreeNode*> ans;
	if (start > end)
	{
		ans.push_back(NULL);
		return ans;
	}
	/*if (start == end)
	{
		TreeNode* temp = new TreeNode(start);
		ans.push_back(temp);
		return ans;
	}*/
	for (int i = start; i <= end; ++i)
	{
		vector<TreeNode*> left_tree = back_generateTrees(start, i - 1);
		vector<TreeNode*> right_tree = back_generateTrees(i + 1, end);
		for (TreeNode* left : left_tree)
		{
			for (TreeNode* right : right_tree)
			{
				TreeNode* root = new TreeNode(i);
				root->left = left;
				root->right = right;
				ans.push_back(root);
			}
		}
	}
	return ans;
}

int CAlgorithmclass::back_numTrees(int start, int end)
{
	int ans = 0;
	if (start >= end)
	{
		return 1;
	}
	for (int i = start; i <= end; ++i)
	{
		int left = back_numTrees(start, i - 1);
		int right = back_numTrees(i + 1, end);
		ans += right * left;
	}
	return ans;
}

bool CAlgorithmclass::back_inInterleave(string& s1, string& s2, string& s3, int i, int j, int k)
{
	if (i == s1.size() && j == s2.size() && k == s3.size())
	{
		return true;
	}
	if (i == s1.size())
	{
		while (j < s2.size())
		{
			if (s3[k] != s2[j])
				return false;
			j++;
			k++;
		}
		return true;
	}
	if (j == s2.size())
	{
		while (i < s1.size())
		{
			if (s3[k] != s1[i])
				return false;
			i++;
			k++;
		}
		return true;
	}
	if (i < s1.size() && s3[k] == s1[i])
	{
		if (back_inInterleave(s1, s2, s3, i + 1, j, k + 1))
			return true;
	}
	if (j < s2.size() && s3[k] == s2[j])
	{
		if (back_inInterleave(s1, s2, s3, i, j + 1, k + 1))
			return true;
	}
	return false;
}

bool CAlgorithmclass::back_isSymmetric(TreeNode * left, TreeNode * right)
{
	if (left == NULL && right == NULL)
		return true;
	if (left == NULL || right == NULL)
		return false;
	if (left->val != right->val)
		return false;
	if (back_isSymmetric(left->left, right->right) && back_isSymmetric(left->right, right->left))
		return true;
	return false;
}

void CAlgorithmclass::back_maxDepth(TreeNode * root, int level)
{
	if (root == NULL)
	{
		m_int = m_int > level ? m_int : level;
		return;
	}
	back_maxDepth(root->left, level + 1);
	back_maxDepth(root->right, level + 1);
}

TreeNode* CAlgorithmclass::back_buildTree(vector<int>& preorder, int l, int r)
{
	if (l == r)
		return NULL;
	TreeNode* root = new TreeNode(preorder[m_int]);
	int i = m_map[preorder[m_int]];
	/*int i = 0;
	for (; i < inorder.size(); i++)
	{
		if (preorder[m_int] == inorder[i])
			break;
	}*/
	m_int++;
	root->left = back_buildTree(preorder, l, i);
	root->right = back_buildTree(preorder, i + 1, r);
	return root;
}

TreeNode * CAlgorithmclass::back_buildTree1(int l, int r)
{
	if (l == r) return nullptr;
	TreeNode* root = new TreeNode(m_intVt[m_int]);
	int i = m_map[m_intVt[m_int]];
	m_int--;
	root->right = back_buildTree1(i + 1, r);
	root->left = back_buildTree1(l, i);
	return root;
}

TreeNode* CAlgorithmclass::back_sortedArrayToBST(vector<int> nums, int l, int r)
{
	if (l == r)
		return NULL;
	int mid = (r + l) / 2;
	TreeNode* root = new TreeNode(nums[mid]);
	root->left = back_sortedArrayToBST(nums, l, mid);
	root->right = back_sortedArrayToBST(nums, mid + 1, r);
	return root;
}

TreeNode * CAlgorithmclass::back_sortedListToBST(ListNode * head, ListNode * end)
{
	if (head == end)
		return NULL;
	ListNode* slow = head;
	ListNode* fast = head;
	while (fast != end && fast->next != end)
	{
		slow = slow->next;
		fast = fast->next->next;
	}
	TreeNode* root = new TreeNode(slow->val);
	root->left = back_sortedListToBST(head, slow);
	root->right = back_sortedListToBST(slow->next, end);
	return root;
}

bool CAlgorithmclass::back_isBalanced(TreeNode * root, int& height)
{
	if (root == NULL)
	{
		height = -1;
		return true;
	}
	int left, right;
	if (back_isBalanced(root->left, left) && back_isBalanced(root->right, right) && abs(left - right) < 2)
	{
		height = left > right ? left : right;
		height++;
		return true;
	}
	return false;
}

int CAlgorithmclass::back_minDepth(TreeNode * root)
{
	if (root == NULL)
		return 0;
	int l = back_minDepth(root->left);
	int r = back_minDepth(root->right);
	if (root->left == NULL || root->right == NULL)
		return l + r + 1;
	return l > r ? r + 1 : l + 1;
}

bool CAlgorithmclass::back_hasPathSum(TreeNode * root, int cur)
{
	if (root == NULL)
		return false;
	cur = cur + root->val;
	if (m_int == cur && root->left == NULL && root->right == NULL)
		return true;
	else
	{
		if (back_hasPathSum(root->left, cur) || back_hasPathSum(root->right, cur))
			return true;
	}
	return false;
}

void CAlgorithmclass::back_PathSum(TreeNode * root, int cur, vector<int>& curVt)
{
	if (root == NULL)
		return;
	cur = cur + root->val;
	curVt.push_back(root->val);
	if (m_int == cur && !root->left && !root->right)
	{
		m_intVtVt.push_back(curVt);
		return;
	}
	if (root->left)
	{
		back_PathSum(root->left, cur, curVt);
		curVt.pop_back();
	}
	if (root->right)
	{
		back_PathSum(root->right, cur, curVt);
		curVt.pop_back();
	}
}

void CAlgorithmclass::back_flatten(TreeNode * root)
{
	if (root == NULL)
		return;
	m_intVt.push_back(root->val);
	back_flatten(root->left);
	back_flatten(root->right);
}

void CAlgorithmclass::back_numDistinct(string& s, string& t, int i, int j)
{
	if (j == t.size()) {
		m_int++;
		return;
	}
	for (int k = i; k < s.size(); ++k)
		if (s[k] == t[j]) back_numDistinct(s, t, k + 1, j + 1);
}

void CAlgorithmclass::back_connect(TreeNode1 * root)
{
	if (root == nullptr || root->left == nullptr) return;
	root->left->next = root->right;
	if (root->next) root->right->next = root->next->left;
	back_connect(root->left);
	back_connect(root->right);
}

void CAlgorithmclass::back_connect2(TreeNode1 * root)
{
	if (root == nullptr || (root->left == nullptr&& root->right == nullptr)) return;
	TreeNode1* temp = root->right;
	if (root->left != nullptr&&root->right != nullptr) root->left->next = root->right;
	else if (root->left) temp = root->left;
	else if (root->right) temp = root->right;
	TreeNode1* cur = root;
	while (cur->next) {
		if (cur->next->left || cur->next->right) {
			if (cur->next->left) temp->next = cur->next->left;
			else if (cur->next->right) temp->next = cur->next->right;
			else temp->next = nullptr;
			break;
		}
		cur = cur->next;
	}
	back_connect2(root->right);
	back_connect2(root->left);
}

int CAlgorithmclass::back_maxPathSum(TreeNode * root)
{
	if (root == nullptr) return 0;
	int leftint = back_maxPathSum(root->left);
	int rightint = back_maxPathSum(root->right);
	m_int = max(m_int, max(max(leftint + root->val, rightint + root->val), max(root->val, leftint + rightint + root->val)));
	return max(root->val, max(leftint + root->val, rightint + root->val));

	/*m_int = max(m_int, root->val);
	m_int = max(m_int, leftint + root->val);
	m_int = max(m_int, rightint + root->val);
	m_int = max(m_int, leftint + rightint + root->val);
	if (leftint <= 0 && rightint <= 0) return root->val;
	else return leftint > rightint ? (leftint + root->val) : (rightint + root->val);*/
}

bool CAlgorithmclass::isconvert(string & s1, string & s2)
{
	//if (s1.size() != s2.size()) return false;
	int count(0);
	for (int i = 0; i < s1.size(); i++) {
		if (s1[i] != s2[i]) count++;
		if (count > 1) return false;
	}
	if (count == 0) return false;
	return true;
}

bool CAlgorithmclass::back_findLadders(unordered_set<string>& words, unordered_set<string>& beginset, unordered_set<string>& endset, bool isend)
{
	if (beginset.empty())
		return false;
	if (beginset.size() > endset.size())
		return back_findLadders(words, endset, beginset, !isend);
	bool ismeet = false;
	unordered_set<string> nextlevel;
	for (auto i : beginset)
	{
		for (int m = 0; m < i.size(); ++m)
		{
			char cm = i[m];
			string new_i = i;
			for (char n = 'a'; n <= 'z'; ++n)
			{
				if (i[m] == n) continue;
				i[m] = n;
				if (words.find(i) == words.end())
					continue;
				nextlevel.insert(i);
				string key = isend ? new_i : i;
				string nextWord = isend ? i : new_i;
				if (endset.find(i) != endset.end())
					ismeet = true;
				m_mmap[key].push_back(nextWord);
			}
			i[m] = cm;
		}
	}
	if (ismeet) return true;
	return back_findLadders(words, nextlevel, endset, isend);
}

void CAlgorithmclass::back_print_findLadders(string & beginWord, string & endWord, vector<string>& strVt)
{
	strVt.push_back(beginWord);
	if (beginWord == endWord) {
		m_strVtVt.push_back(strVt);
		strVt.pop_back();
	}
	if (!m_mmap[beginWord].empty()) {
		for (auto word : m_mmap[beginWord])
		{
			back_print_findLadders(word, endWord, strVt);
		}
	}
	strVt.pop_back();
}

void CAlgorithmclass::back_sumNumbers(TreeNode * root, int cur)
{
	if (root == nullptr)
		return;
	cur = cur * 10 + root->val;
	if (!root->left && !root->right) {
		m_int += cur;
		return;
	}
	back_sumNumbers(root->left, cur);
	back_sumNumbers(root->right, cur);
}

void CAlgorithmclass::back_solve(int i, int j, int row, int col, vector<vector<char>>& board)
{
	board[i][j] = 90;
	for (int k = 0; k < 4; ++k)
	{
		int temp_i = i + m_pairVt[k].first;
		int temp_j = j + m_pairVt[k].second;
		if (temp_i < 0 || temp_j < 0 || temp_i >= row || temp_j >= col || board[temp_i][temp_j] != 79) continue;
		back_solve(temp_i, temp_j, row, col, board);
	}
}

bool CAlgorithmclass::is_partition(string& s, int a, int i)
{
	while (a < i)
		if (s[a++] != s[i--]) return false;
	return true;
}

void CAlgorithmclass::back_partition(string & s, int a, vector<string>& temp, vector<vector<bool>>& dp)
{
	if (a == m_int) {
		m_strVtVt.push_back(temp);
		return;
	}
	for (int i = a; i < m_int; ++i)
	{
		if (/*is_partition(s, a, i)*/dp[a][i]) {
			string s1 = s.substr(a, i - a + 1);
			temp.push_back(s1);
			back_partition(s, i + 1, temp, dp);
			temp.pop_back();
		}
	}
}

bool CAlgorithmclass::back_wordBreak(string & s, int a, vector<bool>& dp)
{
	if (a == s.size())
		return 1;
	if (dp[a])
		return dp[a];
	for (int i = a + 1; i <= s.size(); ++i)
	{
		if (m_setstr.find(s.substr(a, i - a)) != m_setstr.end() && back_wordBreak(s, i, dp))
			return dp[a] = 1;
	}
	return dp[a] = 0;
}

void CAlgorithmclass::back_wordBreak2(string  s, int a, vector<bool>& dp, string s1)
{
	if (a == s.size()) {
		m_strVt.push_back(s1);
		return;
	}
	for (int i = a; i < s.size(); ++i)
	{
		string temp = s.substr(a, i - a + 1);
		if (dp[i + 1] && m_setstr.find(temp) != m_setstr.end()) {
			if (i == s.size() - 1) back_wordBreak2(s, i + 1, dp, s1 + temp);
			else back_wordBreak2(s, i + 1, dp, s1 + temp + " ");
		}
	}
}

ListNode* CAlgorithmclass::reverse_list(ListNode * head)
{
	//递归
	if (!head || !head->next)
		return head;
	ListNode* temp = reverse_list(head->next);
	head->next->next = head;
	head->next = nullptr;
	return temp;
	//迭代 使用栈
	/*stack<ListNode*> s;
	while (head&&head->next){
		s.push(head);
		head = head->next;
	}
	ListNode* ans = head;
	while (!s.empty())
	{
		ListNode* temp = s.top(); s.pop();
		head->next = temp;
		temp->next = nullptr;
		head = head->next;
	}
	return ans;*/
	//迭代 不使用栈
	/*ListNode* pre = nullptr;
	ListNode* cur = head;
	while (cur){
		ListNode* temp = cur->next;
		cur->next = pre;
		pre = cur;
		cur = temp;
	}
	return pre;*/
}

void CAlgorithmclass::SplitString(const string & s, const string & c, vector<string>& v)
{
	std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (string::npos != pos2)//string::npos相当于标志位
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);//第一个参数用来表示要查找的字符  第二个参数是从何处位置开始  默认从0
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}

bool CAlgorithmclass::compair_string(string & s1, string & s2)
{
	return s1 + s2 > s2 + s1;
}

uint32_t CAlgorithmclass::reverseByte(uint32_t _byte, unordered_map<uint32_t, uint32_t>& _map)
{
	if (_map.find(_byte) != _map.end())
		return _map[_byte];
	uint32_t value = (_byte * 0x0202020202 & 0x010884422010) % 1023;
	_map.emplace(_byte, value);
	return value;
}

int CAlgorithmclass::hamming_Weight(uint32_t _byte, unordered_map<uint32_t, int>& _map)
{
	if (_map.find(_byte) != _map.end())
		return _map[_byte];
	int val = 0;
	while (_byte != 0) {
		val++;
		_byte &= _byte - 1;
	}
	_map.emplace(_byte, val);
	return val;
}

bool CAlgorithmclass::intersection_inside(int x1, int y1, int x2, int y2, int xk, int yk)
{
	// 若与 x 轴平行，只需要判断 x 的部分
	// 若与 y 轴平行，只需要判断 y 的部分
	// 若为普通线段，则都要判断
	return (x1 == x2 || (min(x1, x2) <= xk && xk <= max(x1, x2))) && (y1 == y2 || (min(y1, y2) <= yk && yk <= max(y1, y2)));
}

void CAlgorithmclass::intersection_update(vector<double>& ans, double xk, double yk)
{
	// 将一个交点与当前 ans 中的结果进行比较
	// 若更优则替换
	if (!ans.size() || xk < ans[0] || (xk == ans[0] && yk < ans[1])) {
		ans = { xk, yk };
	}
}

void CAlgorithmclass::back_rightSideView(TreeNode * root, int level)
{
	if (!root)
		return;
	if (level >= m_intVt.size())
		m_intVt.push_back(root->val);
	back_rightSideView(root->right, level + 1);
	back_rightSideView(root->left, level + 1);
}

void CAlgorithmclass::back_numIslands(vector<vector<char>>& grid, int i, int j)
{
	grid[i][j] = '0';
	for (int k = 0; k < 4; ++k)
	{
		int temp_i = i + m_pairVt[k].first;
		int temp_j = j + m_pairVt[k].second;
		if (temp_i < 0 || temp_j < 0 || temp_i >= m_row || temp_j >= m_col || grid[temp_i][temp_j] == '0')
			continue;
		back_numIslands(grid, temp_i, temp_j);
	}
}

int CAlgorithmclass::getnext(int n)
{
	int res = 0;
	while (n)
	{
		int yu = n % 10;
		res += yu * yu;
		n /= 10;
	}
	return res;
}

void CAlgorithmclass::back_updateMatrix(vector<vector<int>>& matrix, int i, int j)
{
	for (int k = 0; k < 4; ++k)
	{
		int temp_i = i + m_pairVt[k].first;
		int temp_j = j + m_pairVt[k].second;
		if (temp_i < 0 || temp_j < 0 || temp_i >= m_row || temp_j >= m_col || matrix[temp_i][temp_j] <= matrix[i][j] + 1)
			continue;
		matrix[temp_i][temp_j] = matrix[i][j] + 1;
		back_updateMatrix(matrix, temp_i, temp_j);
	}
}

bool CAlgorithmclass::back_canJump(int a, vector<int>& nums)
{
	if (a >= nums.size() - 1)
		return true;
	for (int i = nums[a]; i > 0; --i)
	{
		if (a + i >= nums.size() - 1 || back_canJump(a + i, nums))
			return true;
	}
	return false;
}

void CAlgorithmclass::back_combinationSum3(vector<vector<int>>& ans, vector<int>& res, int k, int n, int a)
{
	if (n < 0)
		return;
	if (k == 0) {
		if (n == 0)
			ans.push_back(res);
		return;
	}
	for (int i = a + 1; i <= 9; ++i)
	{
		res.push_back(i);
		back_combinationSum3(ans, res, k - 1, n - i, i);
		res.pop_back();
	}
}

int CAlgorithmclass::findnumsTree(TreeNode * root)
{
	if (!root) return 0;
	return findnumsTree(root->left) + findnumsTree(root->right) + 1;
}

bool CAlgorithmclass::back_lowestCommonAncestor(TreeNode * root, TreeNode * p, TreeNode * q)
{
	if (!root)
		return false;
	if (root == p || root == q)
		return true;
	return back_lowestCommonAncestor(root->left, p, q) || back_lowestCommonAncestor(root->right, p, q);
}

int CAlgorithmclass::mergeSort(vector<int>& nums, vector<int>& temp, int l, int r)
{
	if (l >= r)
		return 0;
	int mid = l + (r - l) / 2;
	int count = mergeSort(nums, temp, l, mid) + mergeSort(nums, temp, mid + 1, r);
	int i = l, j = mid + 1, pos = l;
	while (i <= mid && j <= r)
	{
		if (nums[i] > nums[j]) {
			temp[pos] = nums[j];
			count += mid - i + 1;
			++j;
		}
		else {
			temp[pos] = nums[i];
			++i;
		}
		++pos;
	}
	for (int k = i; k <= mid; ++k)
	{
		temp[pos++] = nums[k];
	}
	for (int k = j; k <= r; ++k)
	{
		temp[pos++] = nums[k];
	}
	copy(temp.begin() + l, temp.begin() + r + 1, nums.begin() + l);
	return count;
}

void CAlgorithmclass::back_binaryTreePaths(TreeNode * root, string s)
{
	if (root == nullptr)
		return;
	if (root->left == nullptr&&root->right == nullptr) {
		s += "->" + to_string(root->val);
		m_strVt.push_back(s.substr(2));
		return;
	}
	back_binaryTreePaths(root->left, s + "->" + to_string(root->val));
	back_binaryTreePaths(root->right, s + "->" + to_string(root->val));
}

void CAlgorithmclass::back_permute(vector<int>& nums, vector<int>& temp, vector<bool>& visit)
{
	if (temp.size() == nums.size()) {
		m_intVtVt.push_back(temp);
		return;
	}
	for (int i = 0; i < nums.size(); ++i)
	{
		if (visit[i])
			continue;
		visit[i] = true;
		temp.push_back(nums[i]);
		back_permute(nums, temp, visit);
		temp.pop_back();
		visit[i] = false;
	}
}

void CAlgorithmclass::back_movingCount(vector<vector<bool>>& visit, int i, int j, int k)
{
	visit[i][j] = true;
	for (int m = 0; m < 4; ++m)
	{
		int temp_i = m_pairVt[m].first + i;
		int temp_j = m_pairVt[m].second + j;
		if (temp_i < 0 || temp_j<0 || temp_i >= visit.size() || temp_j >= visit[0].size() || visit[temp_i][temp_j] || help_move(temp_i, temp_j) > k)
			continue;
		m_int++;
		back_movingCount(visit, temp_i, temp_j, k);
	}
}

int CAlgorithmclass::help_move(int i, int j)
{
	int ans = 0;
	while (i) {
		int temp = i % 10;
		ans += temp;
		i /= 10;
	}
	while (j) {
		int temp = j % 10;
		ans += temp;
		j /= 10;
	}
	return ans;
}

ListNode * CAlgorithmclass::mergeLists(vector<ListNode*>& lists, int l, int r)
{
	if (l == r)
		return lists[l];
	int mid = l + (r - l) / 2;//mid = (l+r)>>1;
	ListNode* lnode = mergeLists(lists, l, mid);
	ListNode* rnode = mergeLists(lists, mid + 1, r);
	ListNode* head = new ListNode(0);
	ListNode* cop = head;
	while (lnode&&rnode)
	{
		if (lnode->val < rnode->val) {
			head->next = lnode;
			lnode = lnode->next;
		}
		else {
			head->next = rnode;
			rnode = rnode->next;
		}
		head = head->next;
	}
	head->next = lnode == nullptr ? rnode : lnode;
	head = cop->next;
	cop->next = nullptr;
	delete cop;
	return head;
}

int CAlgorithmclass::back_calculateF(int k, int t)
{
	if (k == 1 || t == 1)
		return t + 1;
	return back_calculateF(k - 1, t - 1) + back_calculateF(k, t - 1);
}

void CAlgorithmclass::back_addOperators(string  num, int a, string resstr, long res, long multi, int target)
{
	if (a >= num.size()) {
		if (res == target)
			m_strVt.push_back(resstr);
		return;
	}
	for (int i = a; i < num.size(); ++i)
	{
		string s = num.substr(a, i - a + 1);
		long n = stol(s);
		if (a == 0) {
			back_addOperators(num, i + 1, resstr + s, n, n, target);
		}
		else {
			back_addOperators(num, i + 1, resstr + '+' + s, res + n, n, target);
			back_addOperators(num, i + 1, resstr + '-' + s, res - n, -n, target);
			back_addOperators(num, i + 1, resstr + '*' + s, res - multi + multi * n, multi * n, target);
		}
		if (n == 0) return;
	}
}

void CAlgorithmclass::back_coinChange(vector<int>& coins, int a, int amount, int cur)
{
	if (amount == 0) {
		m_int = min(m_int, cur);
		return;
	}
	if (a < 0) return;
	for (int i = amount / coins[a]; i >= 0 && i + cur < m_int; --i)
		back_coinChange(coins, a - 1, amount - i * coins[a], cur + i);
}

int CAlgorithmclass::back_diameterOfBinaryTree(TreeNode * root)
{
	if (root == nullptr)
		return 0;
	int l = back_diameterOfBinaryTree(root->left);
	int r = back_diameterOfBinaryTree(root->right);
	m_int = max(m_int, l + r);
	return max(l, r) + 1;
}

void CAlgorithmclass::back_maxAreaOfIsland(vector<vector<int>>& grid, int i, int j)
{
	grid[i][j] = 0;
	for (int k = 0; k < 4; ++k)
	{
		int temp_i = i + m_pairVt[k].first;
		int temp_j = j + m_pairVt[k].second;
		if (temp_i < 0 || temp_j < 0 || temp_i >= m_row || temp_j >= m_col || !grid[temp_i][temp_j])
			continue;
		++m_int;
		back_maxAreaOfIsland(grid, temp_i, temp_j);
	}
}




vector<int> CAlgorithmclass::sortArray(vector<int>& nums)
{
	QuickSort(nums, 0, nums.size() - 1);
	return nums;
}

void CAlgorithmclass::QuickSort(vector<int>& nums, int low, int high)
{
	if (low < high) {
		int l = low, h = high;
		int temp = nums[low];
		while (low < high) {
			while (low < high && nums[high] >= temp)
				--high;
			nums[low] = nums[high];
			while (low < high && nums[low] <= temp)
				++low;
			nums[high] = nums[low];
		}
		nums[low] = temp;
		QuickSort(nums, l, low - 1);
		QuickSort(nums, low + 1, h);
	}

	/*if (l >= r) return;
	int i = l;
	int j = r;
	int x = s[l];
	while (i<j) {
		while (i<j && arr[j] >= x)
			--j;
		if (i<j) arr[i++] = arr[j];
		while (i<j && arr[i]<x)
			++i;
		if (i<j) arr[j--] = arr[i];
	}
	arr[i] = x;
	quickSort(arr, l, i - 1);
	quickSort(arr, i + 1, r);*/
}

void CAlgorithmclass::heapSort(vector<int>& nums)
{
	int len = (int)nums.size() - 1;
	buildMaxHeap(nums, len);
	for (int i = len; i >= 1; --i) {
		swap(nums[i], nums[0]);
		len -= 1;
		maxHeapify(nums, 0, len);
	}
}

vector<int> CAlgorithmclass::MaxKth_heapSort(vector<int>& nums, int k)
{
	vector<int> ans(k);
	int len = (int)nums.size() - 1, sSize = nums.size() - k;
	buildMaxHeap(nums, len);
	for (int i = len; i >= sSize; --i) {
		swap(nums[i], nums[0]);
		len -= 1;
		maxHeapify(nums, 0, len);
	}
	return vector<int>(nums.begin() + nums.size() - k, nums.end());
}

void CAlgorithmclass::buildMaxHeap(vector<int>& nums, int len)
{
	for (int i = len / 2; i >= 0; --i)
		maxHeapify(nums, i, len);
}

void CAlgorithmclass::maxHeapify(vector<int>& nums, int i, int len)
{
	while ((i * 2) + 1 <= len)
	{
		int lson = (i * 2) + 1;
		int rson = (i * 2) + 2;
		int large;
		if (lson <= len && nums[lson] > nums[i])
			large = lson;
		else
			large = i;
		if (rson <= len && nums[rson] > nums[large])
			large = rson;
		if (large != i) {
			swap(nums[i], nums[large]);
			i = large;
		}
		else break;
	}
}

void CAlgorithmclass::MergeSort(vector<int>& nums, int l, int r)
{
	if (l >= r)
		return;
	int mid = l + (r - l) / 2;
	MergeSort(nums, l, mid);
	MergeSort(nums, mid + 1, r);
	int m = l;//全局变量中的m_intVt的指针
	int i = l, j = mid + 1;
	while (i <= mid && j <= r)
	{
		if (nums[i] < nums[j])
			m_intVt[m++] = nums[i++];
		else
			m_intVt[m++] = nums[j++];
	}
	while (i <= mid)
		m_intVt[m++] = nums[i++];
	while (j <= r)
		m_intVt[m++] = nums[j++];
	copy(m_intVt.begin() + l, m_intVt.begin() + r + 1, nums.begin() + l);
}

vector<int> CAlgorithmclass::getLeastNumbers(vector<int>& arr, int k)
{
	QuickSort_Select(arr, 0, arr.size() - 1, k - 1);
	return vector<int>(arr.begin(), arr.begin() + k);
}

void CAlgorithmclass::QuickSort_Select(vector<int>& arr, int start, int end, int k)
{
	if (start == end)
		return;
	int temp = arr[start];
	int i = start, j = end;
	while (i <= j)
	{
		while (i <= j && arr[i] < temp)
			i++;
		while (i <= j && arr[j] > temp)
			j--;
		if (i <= j)
			swap(arr[i++], arr[j--]);
	}
	if (start <= k && k <= j)
		QuickSort_Select(arr, start, j, k);
	else if (i <= k && k <= end)
		QuickSort_Select(arr, i, end, k);
}

bool CAlgorithmclass::KMPstring(const string & s, const string & t)
{
	int sSize = s.size(), tSize = t.size();
	vector<int> next(t.size(), 0);
	for (int i = 1, k = 0; i < tSize; ++i)
	{
		while (k > 0 && t[i] != t[k])
			k = next[k - 1];
		if (t[i] == t[k])
			k++;
		next[i] = k;
	}
	for (int i = 0, q = 0; i < sSize; ++i)
	{
		while (q > 0 && t[q] != s[i])
			q = next[q - 1];
		if (t[q] == s[i])
			q++;
		if (q == tSize)
			return true;
	}
	return false;
}

int CAlgorithmclass::bag1(int N, int M, const vector<int>& w, const vector<int>& v)
{
	//二维空间
	/*vector<vector<int>> dp(N + 1, vector<int>(M + 1, 0));
	for (int i = 1; i <= N; ++i) {
		for (int j = 1; j <= M; ++j) {
			if (j < v[i])
				dp[i][j] = dp[i - 1][j];
			else
				dp[i][j] = max(dp[i - 1][j - v[i]] + w[i],dp[i - 1][j]);
		}
	}
	return dp[N][M];*/
	//一维空间
	vector<int> dp(M + 1, 0);
	for (int i = 1; i <= N; ++i) {
		for (int j = M; j >= 0; --j) {
			if (j >= v[i])
				dp[j] = max(dp[j - v[i]] + w[i], dp[j]);
		}
	}
	int res = 0;
	for (auto& n : dp) res = max(res, n);
	return res;
}

int CAlgorithmclass::bag2(int N, int M, const vector<int>& w, const vector<int>& v)
{
	vector<int> dp(M + 1, 0);
	for (int i = 1; i <= N; ++i) {
		for (int j = 1; j <= M; ++j) {
			if (j >= v[i])
				dp[j] = max(dp[j - v[i]] + w[i], dp[j]);
		}
	}
	int res = 0;
	for (auto& n : dp) res = max(res, n);
	return res;
}

int CAlgorithmclass::bag3(int N, int M, const vector<int>& w, const vector<int>& v, const vector<int>& s)
{
	vector<int> dp(M + 1, 0);
	for (int i = 1; i <= N; ++i) {
		for (int j = M; j >= 0; --j) {
			for (int k = 1; k <= s[i] && k*v[i] <= j; ++k)
				dp[j] = max(dp[j], dp[j - k * v[i]] + k * w[i]);
		}
	}
	return dp[M];
}

int CAlgorithmclass::bag4(int N, int M, const vector<int>& w, const vector<int>& v, const vector<int>& s)
{
	vector<pair<int, int>> goods;
	for (int i = 1; i <= N; ++i) {
		int tmps = s[i];
		for (int k = 1; k <= tmps; k *= 2) {
			tmps -= k;
			goods.push_back({ w[i] * k,v[i] * k });
		}
		if (tmps > 0)
			goods.push_back({ w[i] * tmps,v[i] * tmps });
	}
	vector<int> dp(M + 1, 0);
	for (auto good : goods) {
		for (int j = M; j >= 0; --j) {
			if (j >= good.second)
				dp[j] = max(dp[j], dp[j - good.second] + good.first);
		}
	}
	return dp[M];
}

int CAlgorithmclass::bag5(int N, int M, const vector<int>& w, const vector<int>& v, const vector<int>& s)
{
	vector<good> goods;
	for (int i = 1; i <= N; ++i) {
		if (s[i] == -1)
			goods.push_back({ w[i],v[i],s[i] });
		else if (s[i] == 0)
			goods.push_back({ w[i],v[i],s[i] });
		else {
			int tmp = s[i];
			for (int k = 1; k <= tmp; k *= 2) {
				tmp -= k;
				goods.push_back({ w[i] * k,v[i] * k,-1 });
			}
			if (tmp > 0)
				goods.push_back({ w[i] * tmp,v[i] * tmp,-1 });
		}
	}
	vector<int> dp(M + 1, 0);
	for (auto& g : goods) {
		if (g.s == -1) {
			for (int j = M; j >= g.v; --j) {
				dp[j] = max(dp[j], dp[j - g.v] + g.w);
			}
		}
		else if (g.s == 0) {
			for (int j = g.v; j <= M; ++j) {
				if (j >= g.v)
					dp[j] = max(dp[j], dp[j - g.v] + g.w);
			}
		}
	}
	return dp[M];
}

int CAlgorithmclass::bag6(int N, int V, int M, const vector<int>& v, const vector<int>& m, const vector<int>& w)
{
	vector<vector<int>> dp(V + 1, vector<int>(M + 1, 0));
	for (int i = 1; i <= N; ++i) {
		for (int j = V; j >= v[i]; --j) {
			for (int k = M; k >= m[i]; --k)
				dp[j][k] = max(dp[j][k], dp[j - v[i]][k - m[i]] + w[i]);
		}
	}
	return dp[V][M];
}




int clionGitHub::threeSumClosest(vector<int> &nums, int target) {
	std::sort(nums.begin(), nums.end());
	int res = nums[0] + nums[1] + nums[2];
	for (int i = 0; i < nums.size() - 2; ++i) {
		if (i > 0 && nums[i] == nums[i - 1]) continue;
		int l = i + 1, r = nums.size() - 1;
		while (l < r) {
			int sum = nums[i] + nums[l] + nums[r];
			if (abs(target - res) > abs(target - sum))
				res = sum;
			if (sum > target) {
				r--;
			}
			else if (sum < target) {
				l++;
			}
			else {
				return sum;
			}
		}
	}
	return res;
}

//int clionGitHub::countSubstrings(string s) {
//	/*int sSize = s.size();
//	m_int_clion = 0;
//	for(int i = 0;i<sSize;++i){
//	CountPalin(s,i,i);
//	CountPalin(s,i,i+1);
//	}
//	return m_int_clion;*/
//	int sSize = s.size();
//	clionGitHub::m_int_clion = sSize;
//	vector<vector<bool>> dp(sSize, vector<bool>(sSize, false));
//	for (int i = 0; i<s.size(); ++i) dp[i][i] = true;
//	for (int i = sSize - 1; i >= 0; --i) {
//		for (int j = i + 1; j<sSize; ++j) {
//			if (s[i] == s[j]) {
//				dp[i][j] = j - i == 1 ? true : dp[i + 1][j - 1];
//			}
//			else {
//				dp[i][j] = false;
//			}
//			if (dp[i][j]) clionGitHub::m_int_clion++;
//		}
//	}
//	return clionGitHub::m_int_clion;
//}

//void clionGitHub::CountPalin(const string& s, int l, int r) {
//	while (l <= r&&l >= 0 && r<s.size() && s[l] == s[r]) {
//		clionGitHub::m_int_clion++;
//		l--;
//		r++;
//	}
//}

int clionGitHub::minSubArrayLen(int s, vector<int> &nums) {
	int l = 0, r = 0;
	int res = 0, ans = nums.size() + 1;
	for (; r < nums.size(); ++r) {
		res += nums[r];
		while (res >= s) {
			ans = min(ans, r - l + 1);
			res -= nums[l];
			l++;
		}
	}
	return ans == nums.size() + 1 ? 0 : ans;
}

int clionGitHub::findKthLargest(vector<int> &nums, int k) {
	/*priority_queue<int,vector<int>,greater<int> > q;
	for(int i = 0;i<nums.size();++i){
	if(q.size()<k){
	q.push(nums[i]);
	}
	else if(q.top() < nums[i]){
	q.pop();
	q.push(nums[i]);
	}
	}
	return q.top();*/
	//手动实现最大堆
	int heapSize = nums.size();
	buildMaxHeap(nums, heapSize);
	for (int i = nums.size() - 1; i >= nums.size() - k + 1; --i) {
		swap(nums[0], nums[i]);
		--heapSize;
		maxHeapify(nums, 0, heapSize);
	}
	return nums[0];
}

void clionGitHub::maxHeapify(vector<int> &a, int i, int heapSize) {
	int l = i * 2 + 1, r = i * 2 + 2, largest = i;
	if (l < heapSize && a[l] > a[largest])
		largest = l;
	if (r < heapSize && a[r] > a[largest])
		largest = r;
	if (largest != i) {
		swap(a[largest], a[i]);
		maxHeapify(a, largest, heapSize);
	}
}

void clionGitHub::buildMaxHeap(vector<int> &a, int heapSize) {
	for (int i = heapSize / 2; i >= 0; --i) {
		maxHeapify(a, i, heapSize);
	}
}

int clionGitHub::findLength(vector<int> &A, vector<int> &B) {
	int ans = 0;
	int m = A.size(), n = B.size();
	vector<vector<int>> dp(m, vector<int>(n, 0));
	for (int i = 1; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			if (A[i - 1] == B[j - 1]) {
				dp[i][j] = dp[i - 1][j - 1] + 1;
				ans = max(ans, dp[i][j]);
			}
		}
	}
	return ans;
}

int clionGitHub::kthSmallest(vector<vector<int>> &matrix, int k) {
	//最小堆
	/*int n = matrix.size();
	struct node{
	int x,y,val;
	node(int _x,int _y,int _val):x(_x),y(_y),val(_val) {}
	bool operator > (const node& a) const {
	return a.val < this->val;
	}
	};
	priority_queue<node,vector<node>,greater<node> > q;
	for(int i = 0;i<n;++i){
	q.push(node(i,0,matrix[i][0]));
	}
	for(int i = 0;i< k - 1;++i){
	node now = q.top();
	q.pop();
	if(now.y != n - 1)
	q.push(node(now.x,now.y + 1,matrix[now.x][now.y + 1]));
	}
	return q.top().val;*/
	//二分查找
	int n = matrix.size();
	int l = matrix[0][0], r = matrix[n - 1][n - 1];
	while (l <= r) {
		int mid = l + (r - l) / 2;
		if (check_searchMatrix(matrix, mid, k, n)) {
			r = mid;
		}
		else {
			l = mid + 1;
		}
	}
	return l;
}

bool clionGitHub::searchMatrix(vector<vector<int>> &matrix, int target) {
	if (matrix.size() == 0) return false;
	int m = matrix.size(), n = matrix[0].size();
	int i = m - 1, j = 0;
	while (i >= 0 && j < n) {
		int mid = matrix[i][j];
		if (mid == target) return true;
		else if (mid < target) --i;
		else ++j;
	}
	return false;
}

bool clionGitHub::check_searchMatrix(vector<vector<int>> &matrix, int mid, int k, int n) {
	int num = 0;
	int i = n - 1, j = 0;
	while (i >= 0 && j < matrix.size()) {
		if (matrix[i][j] <= mid) {
			num += i + 1;
			++j;
		}
		else {
			--i;
		}
	}
	return num >= k;
}

TreeNode *clionGitHub::sortedArrayToBST(vector<int> &nums) {
	return back_sortedArrayToBST(nums, 0, nums.size() - 1);
}

TreeNode *clionGitHub::back_sortedArrayToBST(vector<int> &nums, int l, int r) {
	if (l > r) {
		return nullptr;
	}
	int mid = (r + l + 1) / 2;
	TreeNode* node = new TreeNode(nums[mid]);
	node->left = back_sortedArrayToBST(nums, l, mid - 1);
	node->right = back_sortedArrayToBST(nums, mid + 1, r);
	return node;
}

bool clionGitHub::patternMatching(string pattern, string value) {
	string a = "", b = "";
	int aSize = 0, bSize = 0;
	int n = value.size();
	for (const auto& s : pattern) {
		if (s == 'a') aSize++;
		else bSize++;
	}
	if (aSize + bSize == 1 && value.empty()) return true;
	if (aSize == 0 && bSize == 0) {
		return 0;
	}
	else if (aSize == 0) {
		int tmp = n / bSize;
		string temp_b = value.substr(0, tmp);
		for (int k = 0; k < n; k += tmp) {
			if (value.substr(k, tmp) != temp_b) return false;
		}
		return true;
	}
	else if (bSize == 0) {
		int tmp = n / aSize;
		string temp_a = value.substr(0, tmp);
		for (int k = 0; k < n; k += tmp) {
			if (value.substr(k, tmp) != temp_a) return false;
		}
		return true;
	}

	for (int i = 0; i < n / aSize; ++i) {
		int j = (n - aSize * i) / bSize;
		if (aSize*i + bSize * j != n) continue;
		a = value.substr(0, i);
		b = value.substr(i, j);
		if (a == b) continue;
		int pos = 0;
		for (int k = 0; k < pattern.size() && pos < n; ++k) {
			if (pattern[k] == 'a' && value.substr(pos, i) == a) {
				pos += i;
				continue;
			}
			else if (value.substr(pos, j) == b) {
				pos += j;
				continue;
			}
			else break;
		}
		if (pos == n) return true;
	}

	//reverse(pattern.begin(),pattern.end());
	//reverse(value.begin(),value.end());

	for (int i = 0; i < n / bSize; ++i) {
		int j = (n - bSize * i) / aSize;
		if (bSize*i + aSize * j != n) continue;
		b = value.substr(0, i);
		a = value.substr(i, j);
		if (a == b) continue;
		int pos = 0;
		for (int k = 0; k < pattern.size() && pos < n; ++k) {
			if (pattern[k] == 'b' && value.substr(pos, i) == b) {
				pos += i;
				continue;
			}
			else if (value.substr(pos, j) == a) {
				pos += j;
				continue;
			}
			else break;
		}
		if (pos == n) return true;
	}

	return false;
}

int clionGitHub::longestValidParentheses(string s) {
	stack<int> si;
	si.push(-1);
	int ans = 0;
	for (int i = 0; i < s.size(); ++i) {
		if (si.top() == -1) {
			si.push(i);
		}
		else if (s[i] == ')'&&s[si.top()] == '(') {
			si.pop();
			ans = max(ans, i - si.top());
		}
		else
			si.push(i);
	}
	return ans;
}

int clionGitHub::maxScoreSightseeingPair(vector<int> &A) {
	int ans = 0, tmp = A[0];
	for (int i = 1; i < A.size(); ++i) {
		ans = max(ans, tmp + A[i] - i);
		tmp = max(tmp, A[i] + i);
	}
	return ans;
}

int clionGitHub::uniquePathsWithObstacles(vector<vector<int>> &obstacleGrid) {
	int m = obstacleGrid.size();
	int n = obstacleGrid[0].size();
	vector<vector<int>> dp(m, vector<int>(n, 0));
	for (int i = 0; i < m; ++i) {
		if (obstacleGrid[i][0] == 1)
			break;
		dp[i][0] = 1;
	}
	for (int i = 0; i < n; ++i) {
		if (obstacleGrid[0][i] == 1)
			break;
		dp[0][i] = 1;
	}
	for (int i = 1; i < m; ++i) {
		for (int j = 1; j < n; ++j) {
			if (obstacleGrid[i][j] == 1) {
				continue;
			}
			dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
		}
	}
	return dp[m - 1][n - 1];
}

bool clionGitHub::hasPathSum(TreeNode *root, int sum) {
	return back_hasPathSum(root, sum, 0);
}

bool clionGitHub::back_hasPathSum(TreeNode *node, int sum, int ans) {
	if (node == nullptr)
		return false;
	ans += node->val;
	if (!node->left && !node->right&&ans == sum) return true;
	if (back_hasPathSum(node->left, sum, ans) || back_hasPathSum(node->right, sum, ans))
		return true;
	ans -= node->val;
	return false;
}

vector<int> clionGitHub::divingBoard(int shorter, int longer, int k) {
	if (shorter == longer) {
		vector<int> ans;
		if (k == 0) return ans;
		ans.push_back(longer*k);
		return  ans;
	}
	vector<int> ans(k + 1);
	for (int i = 0; i <= k; ++i) {
		int tmp = i * longer + shorter * (k - i);
		ans[i] = tmp;
	}
	return ans;
}

int clionGitHub::respace(vector<string> &dictionary, string sentence) {
	//递归失败
	/*unordered_set<string> aSet;
	int wordLen = 0;
	for(const auto& s:dictionary) {
	wordLen = max(wordLen,(int)s.size());
	aSet.insert(s);
	}
	m_int_clion = sentence.size();
	back_respace(aSet,sentence,wordLen,0,0);
	return m_int_clion;*/
	//动态规划
	int n = sentence.size();
	vector<int> dp(n + 1);
	dp[0] = 0;
	for (int i = 0; i < n; ++i) {
		dp[i + 1] = dp[i] + 1;
		for (auto& word : dictionary) {
			if (word.size() <= i + 1) {
				if (word == sentence.substr(i + 1 - word.size(), word.size()))
					dp[i + 1] = min(dp[i + 1], dp[i + 1 - word.size()]);
			}
		}
	}
	return dp[n];
}

//void clionGitHub::back_respace(unordered_set<string>& dictionary, string& sentence, int wordLen, int x, int num) {
//	if (x >= sentence.size()) {
//		clionGitHub::m_int_clion = min(clionGitHub::m_int_clion, num);
//		return;
//	}
//	for (int i = x + 1; i <= sentence.size(); ++i) {
//		if (i - x > wordLen) {
//			back_respace(dictionary, sentence, wordLen, x + 1, num + 1);
//			return;
//		}
//		string tmp = sentence.substr(x, i - x);
//		if (dictionary.find(tmp) == dictionary.end()) {
//			continue;
//		}
//		back_respace(dictionary, sentence, wordLen, i, num);
//	}
//}

int clionGitHub::maxProfit_freeze(vector<int> &prices) {
	if (prices.empty()) return 0;
	int sSize = prices.size();
	vector<vector<int>> dp(sSize, vector<int>(2, 0));
	dp[0][0] = 0;
	dp[0][1] = -prices[0];
	int dp_pre = 0;//代表dp[i-2][0]
	for (int i = 1; i < sSize; ++i) {
		int tmp = dp[i - 1][0];
		dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
		dp[i][1] = max(dp[i - 1][1], dp_pre - prices[i]);
		dp_pre = tmp;
	}
	return dp[sSize - 1][0];
}

int clionGitHub::maxProfit(vector<int> &prices) {
	if (prices.empty()) return 0;
	int iMin = INT_MAX, iMax = 0;
	for (int i = 0; i < prices.size(); ++i) {
		iMax = max(iMax, prices[i] - iMin);
		iMin = min(iMin, prices[i]);
	}
	return iMax;
	/*if (prices.empty()) return 0;
	int ans(0),profit(0),pre = prices[0];
	for (int i = 1; i < prices.size(); ++i) {
	if (pre > prices[i]) pre = prices[i];
	else  profit = prices[i] - pre;
	ans = ans > profit ? ans : profit;
	}
	return ans;*/
}

int clionGitHub::maxProfit2(vector<int> &prices) {
	if (prices.empty()) return 0;
	int sSize = prices.size();
	vector<vector<int>> dp(sSize, vector<int>(2, 0));
	dp[0][0] = 0;
	dp[0][1] = -prices[0];
	for (int i = 1; i < sSize; ++i) {
		dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
		dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
	}
	return dp[sSize - 1][0];
}

int clionGitHub::maxProfit3(vector<int> &prices) {
	if (prices.empty()) return 0;
	int sSize = prices.size();
	//vector<vector<vector<int>>> dp(sSize,vector<vector<int>>(2,vector<int>(2,0)));
	int dp_1_0 = 0;
	int dp_1_1 = -prices[0];
	int dp_2_0 = 0;
	int dp_2_1 = INT_MIN;
	for (int i = 1; i < sSize; ++i) {
		dp_2_0 = max(dp_2_0, dp_2_1 + prices[i]);
		dp_2_1 = max(dp_2_1, dp_1_0 - prices[i]);
		dp_1_0 = max(dp_1_0, dp_1_1 + prices[i]);
		dp_1_1 = max(dp_1_1, -prices[i]);
	}
	return max(dp_1_0, dp_2_0);
}

int clionGitHub::maxProfit4(int k, vector<int> &prices) {
	if (prices.empty()) return 0;
	if (k >= prices.size() / 2)
		return maxProfit3(prices);
	int sSize = prices.size();
	vector<vector<vector<int>>> dp(sSize, vector<vector<int>>(k + 1, vector<int>(2, 0)));
	for (int i = 0; i < sSize; ++i) {
		for (int j = k; j > 0; --j) {
			if (i == 0) {
				dp[0][j][0] = 0;
				dp[0][j][1] = -prices[0];
				continue;
			}
			dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
			dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);
		}
	}
	return dp[sSize - 1][k][0];
}

vector<int> clionGitHub::countSmaller(vector<int> &nums) {
	int sSize = nums.size();
	vector<int> res(sSize, 0);
	vector<int> index(sSize, 0);
	for (int i = 0; i < sSize; ++i) index[i] = i;
	mergeCountSmaller(nums, index, res, 0, sSize - 1);
	return res;
}

void clionGitHub::mergeCountSmaller(vector<int> &nums, vector<int> &index, vector<int> &res, int l, int r) {
	if (l >= r) {
		return;
	}
	int mid = l + (r - l) / 2;
	mergeCountSmaller(nums, index, res, l, mid);
	mergeCountSmaller(nums, index, res, mid + 1, r);
	vector<int> tmp(r - l + 1);
	int i = l, j = mid + 1;
	int pos = l;
	while (i <= mid && j <= r) {
		if (nums[index[j]] < nums[index[i]]) {
			res[index[i]] += r - j + 1;
			tmp[pos] = index[i];
			i++;
		}
		else {
			tmp[pos] = index[j];
			j++;
		}
		pos++;
	}
	while (i <= mid) {
		tmp[pos++] = index[i++];
	}
	while (j <= r) {
		tmp[pos++] = index[j++];
	}
	std::copy(tmp.begin(), tmp.end(), index.begin() + l);
}

int clionGitHub::calculateMinimumHP(vector<vector<int>> &dungeon) {
	int row = dungeon.size(), col = dungeon[0].size();
	dungeon[row - 1][col - 1] = -dungeon[row - 1][col - 1] < 0 ? 0 : -dungeon[row - 1][col - 1];
	for (int i = col - 2; i >= 0; --i)
		dungeon[row - 1][i] = max(0, dungeon[row - 1][i + 1]) - dungeon[row - 1][i];
	for (int i = row - 2; i >= 0; --i)
		dungeon[i][col - 1] = max(0, dungeon[i + 1][col - 1]) - dungeon[i][col - 1];
	for (int i = row - 2; i >= 0; --i) {
		for (int j = col - 2; j >= 0; --j)
			dungeon[i][j] = max(0, min(dungeon[i][j + 1], dungeon[i + 1][j])) - dungeon[i][j];
	}
	return dungeon[0][0] < 0 ? 1 : dungeon[0][0] + 1;
}

int clionGitHub::numIdenticalPairs(vector<int> &nums) {
	int sSize = nums.size();
	vector<vector<int>> tmp(101);
	for (int i = 0; i < sSize; ++i) {
		tmp[nums[i]].push_back(i);
	}
	int ans = 0;
	for (int i = 0; i <= 100; ++i) {
		if (tmp[i].empty()) continue;
		int len = tmp[i].size();
		ans += len * (len - 1) / 2;
	}
	return ans;
}

int clionGitHub::numSub(string s) {
	if (s.empty()) return 0;
	vector<long long> dp(s.size() + 1);
	dp[0] = 0;
	int index = 0;
	for (int i = 1; i <= s.size(); ++i) {
		if (s[i - 1] != '1') {
			dp[i] = dp[i - 1];
			index = i;
		}
		else {
			int n = i - index;
			dp[i] = dp[i - 1] + n;
		}
	}
	return dp[s.size()] % (1000000007);
}

double clionGitHub::maxProbability(int n, vector<vector<int>> &edges, vector<double> &succProb, int start, int end) {
	/*vector<vector<double>> adjacency(n,vector<double>(n,0.0));
	//构建邻接图
	for(int i = 0;i<edges.size();++i){
	adjacency[edges[i][0]][edges[i][1]] = succProb[i];
	adjacency[edges[i][1]][edges[i][0]] = succProb[i];
	}
	int count = 1;
	vector<std::pair<double,bool>> dis(n);
	for(int i = 0;i<n;++i){
	dis[i].first = adjacency[start][i];
	dis[i].second = false;
	}
	dis[start].first = 1.0;
	dis[start].second = true;
	while (count != n){
	double iMax = 0.0;
	int tmp = -1;
	for(int i = 0;i<n;++i){
	if(!dis[i].second && dis[i].first > iMax){
	iMax = dis[i].first;
	tmp = i;
	}
	}
	dis[tmp].second = true;
	++count;
	for(int i = 0;i<n;++i){
	if(!dis[i].second && dis[tmp].first * adjacency[tmp][i] > dis[i].first){
	dis[i].first = dis[tmp].first * adjacency[tmp][i];
	}
	}
	}
	return dis[end].first;*/

	vector<vector<pair<double, int>>> graph(n, vector<pair<double, int>>());
	for (int i = 0; i < edges.size(); ++i) {
		auto e = edges[i];
		graph[e[0]].push_back({ succProb[i],e[1] });
		graph[e[1]].push_back({ succProb[i],e[0] });
	}
	vector<int> visited(n, 0);
	priority_queue<pair<double, int>> q;
	q.push({ 1,start });
	while (!q.empty()) {
		auto p = q.top();
		q.pop();
		auto curProb = p.first;
		auto curPos = p.second;
		if (visited[curPos]) continue;
		visited[curPos] = 1;
		if (curPos == end) return curProb;
		for (auto next : graph[curPos]) {
			double nextProb = next.first;
			int nextPos = next.second;
			if (visited[nextPos]) continue;
			q.push({ curProb*nextProb,nextPos });
		}
	}
	return 0;
}
