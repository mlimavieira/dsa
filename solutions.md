# LeetCode Solutions Summary

This markdown document contains solution strategies for selected LeetCode problems. Each entry includes a problem name, number, difficulty, and a brief solution sketch or approach.

---

### ✅ 1. Two Sum (#1, Easy)
Use a HashMap to store the complement of each number while iterating.

```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        if (map.containsKey(complement)) {
            return new int[]{map.get(complement), i};
        }
        map.put(nums[i], i);
    }
    return new int[0];
}
```
---
### ✅ 540. Single Element in a Sorted Array (Medium)
Use binary search on even indices to find the unique element.

```java
public int singleNonDuplicate(int[] nums) {
    int left = 0, right = nums.length - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (mid % 2 == 1) mid--; // make sure mid is even
        if (nums[mid] == nums[mid + 1]) {
            left = mid + 2;
        } else {
            right = mid;
        }
    }
    return nums[left];
}
```
---

### ✅ 14. Longest Common Prefix (#14, Easy)
Sort the array and compare first and last string characters.

```java
    public String longestCommonPrefix(final String[] strs) {
        if(strs == null || strs.length == 0) {
            return "";
        }
        String prefix = getShortestString(strs);

        for(final String s : strs){

            while(!s.startsWith(prefix)) {
                prefix = prefix.substring(0, prefix.length() - 1);
                if(prefix.isEmpty()) {
                    return "";
                }
            }
        }
        return prefix;
    }

```

---

### ✅ 146. LRU Cache (#146, Medium)
Use a LinkedHashMap or implement with HashMap + Doubly Linked List.

```java
class LRUCache {

    private class Node {
        int key, value;
        Node prev, next;
        Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    private final Map<Integer, Node> map;
    private final int capacity;
    private final Node head, tail;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        map = new HashMap<>();
        head = new Node(0, 0); // dummy head
        tail = new Node(0, 0); // dummy tail
        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        if (!map.containsKey(key)) return -1;
        Node node = map.get(key);
        remove(node);
        insertToFront(node);
        return node.value;
    }

    public void put(int key, int value) {
        if (map.containsKey(key)) {
            remove(map.get(key));
        }
        Node node = new Node(key, value);
        insertToFront(node);
        map.put(key, node);
        if (map.size() > capacity) {
            Node lru = tail.prev;
            remove(lru);
            map.remove(lru.key);
        }
    }

    private void remove(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void insertToFront(Node node) {
        node.next = head.next;
        node.prev = head;
        head.next.prev = node;
        head.next = node;
    }
}
```

---

### ✅ 53. Maximum Subarray (#53, Medium)
Kadane’s Algorithm – track current sum and global max.

```java
public int maxSubArray(int[] nums) {
    int maxSum = nums[0], currSum = nums[0];
    for (int i = 1; i < nums.length; i++) {
        currSum = Math.max(nums[i], currSum + nums[i]);
        maxSum = Math.max(maxSum, currSum);
    }
    return maxSum;
}
```

---

### ✅ 70. Climbing Stairs (#70, Easy)
Use Fibonacci-style DP: dp[n] = dp[n-1] + dp[n-2].

```java
public int climbStairs(int n) {
    if (n <= 2) return n;
    int a = 1, b = 2;
    for (int i = 3; i <= n; i++) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}
```

---

### ✅ 88. Merge Sorted Array (#88, Easy)
Two pointers from end; merge into nums1 backwards.

```java
    public void merge(final int[] nums1, final int m, final int[] nums2, final int n) {
        final int[] auxNums1 = Arrays.copyOf(nums1, nums1.length);
        int idxNums1 = 0;
        int idxNums2 = 0;
        for (int i = 0; i < n + m; i++) {
            if (idxNums2 >= n || (idxNums1 < m && auxNums1[idxNums1] < nums2[idxNums2])) {
                nums1[i] = auxNums1[idxNums1++];
                idxNums1++;
            } else {
                nums1[i] = auxNums1[idxNums2++];
                idxNums2++;
            }
        }
    }
```

---

### ✅ 13. Roman to Integer (#13, Easy)
Map values and subtract when a smaller value precedes a larger.

```java
public int romanToInt(String s) {
    Map<Character, Integer> map = Map.of(
        'I', 1, 'V', 5, 'X', 10, 'L', 50,
        'C', 100, 'D', 500, 'M', 1000
    );
    int total = 0;
    for (int i = 0; i < s.length(); i++) {
        int curr = map.get(s.charAt(i));
        int next = (i + 1 < s.length()) ? map.get(s.charAt(i + 1)) : 0;
        total += (curr < next) ? -curr : curr;
    }
    return total;
}
```

---

### ✅ 3. Longest Substring Without Repeating Characters (#3, Medium)
Sliding window with a HashMap or HashSet.

```java
public int lengthOfLongestSubstring(String s) {
    Set<Character> set = new HashSet<>();
    int left = 0, maxLen = 0;
    for (int right = 0; right < s.length(); right++) {
        while (!set.add(s.charAt(right))) {
            set.remove(s.charAt(left++));
        }
        maxLen = Math.max(maxLen, right - left + 1);
    }
    return maxLen;
}
```

---

### ✅ 136. Single Number (#136, Easy)
XOR all elements: duplicates cancel out.

```java
public int singleNumber(int[] nums) {
    int result = 0;
    for (int num : nums) {
        result ^= num;
    }
    return result;
}
```

---

### ✅ 26. Remove Duplicates from Sorted Array (#26, Easy)
Two pointers – overwrite duplicates in place.

```java
public int removeDuplicates(int[] nums) {
    int i = 0;
    for (int j = 1; j < nums.length; j++) {
        if (nums[j] != nums[i]) {
            nums[++i] = nums[j];
        }
    }
    return i + 1;
}
```

---

### ✅ 283. Move Zeroes (#283, Easy)
Two pointers – move non-zero forward, then fill remaining with zero.

```java
public void moveZeroes(int[] nums) {
    int index = 0;
    for (int num : nums) {
        if (num != 0) nums[index++] = num;
    }
    while (index < nums.length) {
        nums[index++] = 0;
    }
}
```

---

### ✅ 49. Group Anagrams (#49, Medium)
Use a HashMap with sorted string as key.

```java
public List<List<String>> groupAnagrams(String[] strs) {
    Map<String, List<String>> map = new HashMap<>();
    for (String s : strs) {
        char[] chars = s.toCharArray();
        Arrays.sort(chars);
        String key = new String(chars);
        map.computeIfAbsent(key, k -> new ArrayList<>()).add(s);
    }
    return new ArrayList<>(map.values());
}
```

---

### ✅ 50. Pow(x, n) (#50, Medium)
Binary exponentiation – fast power method.

```java
public double myPow(double x, int n) {
    long N = n;
    if (N < 0) {
        x = 1 / x;
        N = -N;
    }
    double result = 1;
    while (N > 0) {
        if (N % 2 == 1) result *= x;
        x *= x;
        N /= 2;
    }
    return result;
}
```

---

### ✅ 56. Merge Intervals (#56, Medium)
Sort by start; merge overlapping by comparing end points.

```java
public int[][] merge(int[][] intervals) {
    Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
    List<int[]> result = new ArrayList<>();
    int[] current = intervals[0];
    for (int i = 1; i < intervals.length; i++) {
        if (intervals[i][0] <= current[1]) {
            current[1] = Math.max(current[1], intervals[i][1]);
        } else {
            result.add(current);
            current = intervals[i];
        }
    }
    result.add(current);
    return result.toArray(new int[result.size()][]);
}
```
---

### ✅ 198. House Robber (#198, Medium)
DP with the recurrence: dp[i] = max(dp[i-1], dp[i-2] + nums[i]).

```java
public int rob(int[] nums) {
    if (nums.length == 0) return 0;
    if (nums.length == 1) return nums[0];
    int first = nums[0], second = Math.max(nums[0], nums[1]);
    for (int i = 2; i < nums.length; i++) {
        int temp = Math.max(second, first + nums[i]);
        first = second;
        second = temp;
    }
    return second;
}
```
---

### ✅ 200. Number of Islands (#200, Medium)
DFS or BFS – mark visited land ('1') during traversal.

```java
public int numIslands(char[][] grid) {
    int count = 0;
    for (int i = 0; i < grid.length; i++) {
        for (int j = 0; j < grid[0].length; j++) {
            if (grid[i][j] == '1') {
                dfs(grid, i, j);
                count++;
            }
        }
    }
    return count;
}

private void dfs(char[][] grid, int i, int j) {
    if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] != '1') return;
    grid[i][j] = '0';
    dfs(grid, i + 1, j);
    dfs(grid, i - 1, j);
    dfs(grid, i, j + 1);
    dfs(grid, i, j - 1);
}
```
---

### ✅ 206. Reverse Linked List (#206, Easy)
Iterative or recursive pointer reversal.

```java
public ListNode reverseList(ListNode head) {
    ListNode prev = null;
    while (head != null) {
        ListNode next = head.next;
        head.next = prev;
        prev = head;
        head = next;
    }
    return prev;
}
```
---

### ✅ 215. Kth Largest Element in an Array (#215, Medium)
Use a Min Heap of size k or Quickselect.

```java
public int findKthLargest(int[] nums, int k) {
    PriorityQueue<Integer> heap = new PriorityQueue<>();
    for (int num : nums) {
        heap.offer(num);
        if (heap.size() > k) {
            heap.poll();
        }
    }
    return heap.peek();
}
```



### 🟡 2. Add Two Numbers (#2, Medium)
Linked list traversal with carry handling.

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(0), curr = dummy;
    int carry = 0;
    while (l1 != null || l2 != null || carry != 0) {
        int sum = carry;
        if (l1 != null) {
            sum += l1.val;
            l1 = l1.next;
        }
        if (l2 != null) {
            sum += l2.val;
            l2 = l2.next;
        }
        carry = sum / 10;
        curr.next = new ListNode(sum % 10);
        curr = curr.next;
    }
    return dummy.next;
}
```
---

### 🟡 121. Best Time to Buy and Sell Stock (#121, Easy)
Track the min price and max profit while iterating.

```java
public int maxProfit(int[] prices) {
    int minPrice = Integer.MAX_VALUE, maxProfit = 0;
    for (int price : prices) {
        if (price < minPrice) minPrice = price;
        else maxProfit = Math.max(maxProfit, price - minPrice);
    }
    return maxProfit;
}
```
---

### 🟡 169. Majority Element (#169, Easy)
Boyer-Moore Voting Algorithm.

```java
public int majorityElement(int[] nums) {
    int count = 0, candidate = 0;
    for (int num : nums) {
        if (count == 0) candidate = num;
        count += (num == candidate) ? 1 : -1;
    }
    return candidate;
}
```
---

### 🟡 234. Palindrome Linked List (#234, Easy)
Reverse second half and compare with first.

```java
public boolean isPalindrome(ListNode head) {
    ListNode slow = head, fast = head;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    ListNode secondHalf = reverse(slow);
    ListNode firstHalf = head;
    while (secondHalf != null) {
        if (firstHalf.val != secondHalf.val) return false;
        firstHalf = firstHalf.next;
        secondHalf = secondHalf.next;
    }
    return true;
}

private ListNode reverse(ListNode head) {
    ListNode prev = null;
    while (head != null) {
        ListNode next = head.next;
        head.next = prev;
        prev = head;
        head = next;
    }
    return prev;
}
```
---

### 🟡 104. Maximum Depth of Binary Tree (#104, Easy)
DFS to compute depth recursively.

```java
public int maxDepth(TreeNode root) {
    if (root == null) return 0;
    return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
}
```
---

### 🟡 22. Generate Parentheses (#22, Medium)
Backtracking with open/close count.

```java
public List<String> generateParenthesis(int n) {
    List<String> result = new ArrayList<>();
    backtrack(result, "", 0, 0, n);
    return result;
}

private void backtrack(List<String> result, String curr, int open, int close, int max) {
    if (curr.length() == max * 2) {
        result.add(curr);
        return;
    }
    if (open < max) backtrack(result, curr + "(", open + 1, close, max);
    if (close < open) backtrack(result, curr + ")", open, close + 1, max);
}
```
---

### 🟡 21. Merge Two Sorted Lists (#21, Easy)
Iterative comparison using dummy node.

```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(0), current = dummy;
    while (l1 != null && l2 != null) {
        if (l1.val < l2.val) {
            current.next = l1;
            l1 = l1.next;
        } else {
            current.next = l2;
            l2 = l2.next;
        }
        current = current.next;
    }
    current.next = (l1 != null) ? l1 : l2;
    return dummy.next;
}
```
---

### 🟥 15. 3Sum (#15, Medium)
Sort and use two-pointer technique.

```java
public List<List<Integer>> threeSum(int[] nums) {
    Arrays.sort(nums);
    List<List<Integer>> result = new ArrayList<>();
    for (int i = 0; i < nums.length - 2; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;
        int left = i + 1, right = nums.length - 1;
        while (left < right) {
            int sum = nums[i] + nums[left] + nums[right];
            if (sum == 0) {
                result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                while (left < right && nums[left] == nums[left + 1]) left++;
                while (left < right && nums[right] == nums[right - 1]) right--;
                left++; right--;
            } else if (sum < 0) left++;
            else right--;
        }
    }
    return result;
}
```
---

### 🟥 23. Merge k Sorted Lists (#23, Hard)
Use a min-heap (PriorityQueue).

```java
public ListNode mergeKLists(ListNode[] lists) {
    PriorityQueue<ListNode> pq = new PriorityQueue<>(Comparator.comparingInt(n -> n.val));
    for (ListNode node : lists) {
        if (node != null) pq.add(node);
    }
    ListNode dummy = new ListNode(0), current = dummy;
    while (!pq.isEmpty()) {
        ListNode node = pq.poll();
        current.next = node;
        current = node;
        if (node.next != null) pq.add(node.next);
    }
    return dummy.next;
}
```
---

### 🟥 42. Trapping Rain Water (#42, Hard)
Use two-pointer approach to find bounded water.

```java
public int trap(int[] height) {
    int left = 0, right = height.length - 1, leftMax = 0, rightMax = 0, res = 0;
    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] >= leftMax) leftMax = height[left];
            else res += leftMax - height[left];
            left++;
        } else {
            if (height[right] >= rightMax) rightMax = height[right];
            else res += rightMax - height[right];
            right--;
        }
    }
    return res;
}
```
---

### 🟥 127. Word Ladder (#127, Hard)
Use BFS to find the shortest transformation sequence.

```java
public int ladderLength(String beginWord, String endWord, List<String> wordList) {
    Set<String> wordSet = new HashSet<>(wordList);
    if (!wordSet.contains(endWord)) return 0;
    Queue<String> queue = new LinkedList<>();
    queue.offer(beginWord);
    int level = 1;
    while (!queue.isEmpty()) {
        int size = queue.size();
        for (int i = 0; i < size; i++) {
            char[] word = queue.poll().toCharArray();
            for (int j = 0; j < word.length; j++) {
                char original = word[j];
                for (char c = 'a'; c <= 'z'; c++) {
                    word[j] = c;
                    String next = new String(word);
                    if (next.equals(endWord)) return level + 1;
                    if (wordSet.remove(next)) queue.offer(next);
                }
                word[j] = original;
            }
        }
        level++;
    }
    return 0;
}
```
---

### 🟥 297. Serialize and Deserialize Binary Tree (#297, Hard)
Use preorder traversal for both serialization and deserialization.

```java
public class Codec {
    public String serialize(TreeNode root) {
        if (root == null) return "#";
        return root.val + "," + serialize(root.left) + "," + serialize(root.right);
    }

    public TreeNode deserialize(String data) {
        Queue<String> nodes = new LinkedList<>(Arrays.asList(data.split(",")));
        return buildTree(nodes);
    }

    private TreeNode buildTree(Queue<String> nodes) {
        String val = nodes.poll();
        if (val.equals("#")) return null;
        TreeNode root = new TreeNode(Integer.parseInt(val));
        root.left = buildTree(nodes);
        root.right = buildTree(nodes);
        return root;
    }
}
```
---


### 🟢 455. Assign Cookies (#455, Easy)
Sort greed factor and cookie size arrays. Match smallest cookie that satisfies each child.

```java
public int findContentChildren(int[] g, int[] s) {
    Arrays.sort(g);
    Arrays.sort(s);
    int i = 0, j = 0;
    while (i < g.length && j < s.length) {
        if (s[j] >= g[i]) i++;
        j++;
    }
    return i;
}
```
---

### 🟢 645. Set Mismatch (#645, Easy)
Use a frequency array to detect the duplicate and missing.

```java
public int[] findErrorNums(int[] nums) {
    int[] freq = new int[nums.length + 1];
    for (int num : nums) freq[num]++;
    int dup = -1, miss = -1;
    for (int i = 1; i < freq.length; i++) {
        if (freq[i] == 2) dup = i;
        if (freq[i] == 0) miss = i;
    }
    return new int[]{dup, miss};
}
```
---

### 🟢 1768. Merge Strings Alternately (#1768, Easy)
Alternate between characters from both strings.

```java
public String mergeAlternately(String word1, String word2) {
    StringBuilder sb = new StringBuilder();
    int i = 0, j = 0;
    while (i < word1.length() || j < word2.length()) {
        if (i < word1.length()) sb.append(word1.charAt(i++));
        if (j < word2.length()) sb.append(word2.charAt(j++));
    }
    return sb.toString();
}
```
---

### 🟡 48. Rotate Image (#48, Medium)
Transpose then reverse each row.

```java
public void rotate(int[][] matrix) {
    int n = matrix.length;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            int temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
    for (int[] row : matrix) {
        for (int i = 0, j = row.length - 1; i < j; i++, j--) {
            int temp = row[i];
            row[i] = row[j];
            row[j] = temp;
        }
    }
}
```
---

### 🟡 235. Lowest Common Ancestor of a BST (#235, Medium)
Recursive BST traversal.

```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (p.val < root.val && q.val < root.val) {
        return lowestCommonAncestor(root.left, p, q);
    } else if (p.val > root.val && q.val > root.val) {
        return lowestCommonAncestor(root.right, p, q);
    } else {
        return root;
    }
}
```
---

### 🟡 875. Koko Eating Bananas (#875, Medium)
Binary search on possible eating speeds.

```java
public int minEatingSpeed(int[] piles, int h) {
    int low = 1, high = Arrays.stream(piles).max().getAsInt();
    while (low < high) {
        int mid = (low + high) / 2;
        int time = 0;
        for (int pile : piles) time += (pile + mid - 1) / mid;
        if (time > h) low = mid + 1;
        else high = mid;
    }
    return low;
}
```
---

### 🟡 901. Online Stock Span (#901, Medium)
Use a stack to store previous prices and spans.

```java
class StockSpanner {
    Stack<int[]> stack = new Stack<>();
    public int next(int price) {
        int span = 1;
        while (!stack.isEmpty() && stack.peek()[0] <= price) {
            span += stack.pop()[1];
        }
        stack.push(new int[]{price, span});
        return span;
    }
}
```
---

### 🟡 1834. Single-Threaded CPU (#1834, Medium)
Use a priority queue for task scheduling by time.

```java
public int[] getOrder(int[][] tasks) {
    int n = tasks.length;
    int[][] indexed = new int[n][3];
    for (int i = 0; i < n; i++) {
        indexed[i] = new int[]{tasks[i][0], tasks[i][1], i};
    }
    Arrays.sort(indexed, Comparator.comparingInt(a -> a[0]));
    PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] == b[1] ? a[2] - b[2] : a[1] - b[1]);

    int[] result = new int[n];
    int time = 0, i = 0, idx = 0;
    while (idx < n) {
        while (i < n && indexed[i][0] <= time) pq.offer(indexed[i++]);
        if (pq.isEmpty()) {
            time = indexed[i][0];
        } else {
            int[] task = pq.poll();
            result[idx++] = task[2];
            time += task[1];
        }
    }
    return result;
}
```
---

### 🟡 2874. Maximum Value of an Ordered Triplet II (#2874, Medium)
Track max prefix and suffix to find maximum ordered triplet.

```java
public int maxValueOfTriplet(int[] nums) {
    int max = 0, n = nums.length;
    TreeSet<Integer> left = new TreeSet<>();
    int[] rightMax = new int[n];
    rightMax[n - 1] = nums[n - 1];
    for (int i = n - 2; i >= 0; i--) rightMax[i] = Math.max(rightMax[i + 1], nums[i]);
    for (int i = 1; i < n - 1; i++) {
        left.add(nums[i - 1]);
        Integer smaller = left.lower(nums[i]);
        if (smaller != null && rightMax[i + 1] > nums[i]) {
            max = Math.max(max, smaller * nums[i] * rightMax[i + 1]);
        }
    }
    return max;
}
```
---

### 🟢 1863. Sum of All Subset XOR Totals (#1863, Easy)
Backtracking to explore all subsets and sum their XORs.

```java
public int subsetXORSum(int[] nums) {
    return dfs(nums, 0, 0);
}

private int dfs(int[] nums, int index, int currentXOR) {
    if (index == nums.length) return currentXOR;
    return dfs(nums, index + 1, currentXOR ^ nums[index]) + dfs(nums, index + 1, currentXOR);
}
```
---



