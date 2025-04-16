# Dynamic Programming (DP) Templates

This document includes essential templates for solving common Dynamic Programming problems, each with a description, use case, and a LeetCode example.

---

## 🧱 1. 0/1 Knapsack (Classic DP Table)
**When to use:** Selecting items with weights and values without exceeding a capacity.  
**LeetCode Example:** [Subset Sum - LeetCode #416](https://leetcode.com/problems/partition-equal-subset-sum/)

```java
boolean canPartition(int[] nums) {
    int sum = Arrays.stream(nums).sum();
    if (sum % 2 != 0) return false;
    int target = sum / 2;
    boolean[] dp = new boolean[target + 1];
    dp[0] = true;
    for (int num : nums) {
        for (int j = target; j >= num; j--) {
            dp[j] = dp[j] || dp[j - num];
        }
    }
    return dp[target];
}
```

---

## 🔁 2. Fibonacci / Linear DP
**When to use:** Problems where the answer depends on the last one or two subproblems.  
**LeetCode Example:** [Climbing Stairs - LeetCode #70](https://leetcode.com/problems/climbing-stairs/)

```java
int climbStairs(int n) {
    if (n <= 2) return n;
    int first = 1, second = 2;
    for (int i = 3; i <= n; i++) {
        int third = first + second;
        first = second;
        second = third;
    }
    return second;
}
```

---

## 📦 3. House Robber (Non-Adjacent Selection)
**When to use:** Cannot pick adjacent items.  
**LeetCode Example:** [House Robber - LeetCode #198](https://leetcode.com/problems/house-robber/)

```java
int rob(int[] nums) {
    if (nums.length == 0) return 0;
    if (nums.length == 1) return nums[0];
    int first = nums[0], second = Math.max(nums[0], nums[1]);
    for (int i = 2; i < nums.length; i++) {
        int current = Math.max(second, first + nums[i]);
        first = second;
        second = current;
    }
    return second;
}
```

---

## 🧠 4. Longest Increasing Subsequence (LIS)
**When to use:** Problems involving ordering or sequence optimization.  
**LeetCode Example:** [Longest Increasing Subsequence - LeetCode #300](https://leetcode.com/problems/longest-increasing-subsequence/)

```java
int lengthOfLIS(int[] nums) {
    int[] dp = new int[nums.length];
    int len = 0;
    for (int num : nums) {
        int i = Arrays.binarySearch(dp, 0, len, num);
        if (i < 0) i = -(i + 1);
        dp[i] = num;
        if (i == len) len++;
    }
    return len;
}
```

---

## ✂️ 5. Edit Distance
**When to use:** String transformation problems.  
**LeetCode Example:** [Edit Distance - LeetCode #72](https://leetcode.com/problems/edit-distance/)

```java
int minDistance(String word1, String word2) {
    int m = word1.length(), n = word2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1.charAt(i - 1) == word2.charAt(j - 1))
                dp[i][j] = dp[i - 1][j - 1];
            else
                dp[i][j] = 1 + Math.min(dp[i - 1][j - 1],
                                 Math.min(dp[i - 1][j], dp[i][j - 1]));
        }
    }
    return dp[m][n];
}
```
