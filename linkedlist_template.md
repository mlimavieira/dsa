# Linked List Algorithm Templates

This document includes common algorithm templates for solving Linked List problems, each with a description of when to use it and a LeetCode problem example.

---

## ✅ 1. Traverse a Linked List

**When to use:** For iterating through nodes.  
**LeetCode Example:** [Convert Binary Number in a Linked List to Integer - LeetCode #1290](https://leetcode.com/problems/convert-binary-number-in-a-linked-list-to-integer/)


```java
void traverse(ListNode head) {
    while (head != null) {
        System.out.println(head.val);
        head = head.next;
    }
}
```

---

## 🔁 2. Reverse a Linked List (Iterative)

**When to use:** Reversing singly linked list.  
**LeetCode Example:** [Reverse Linked List - LeetCode #206](https://leetcode.com/problems/reverse-linked-list/)


```java
ListNode reverseList(ListNode head) {
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

## 🔁 3. Reverse a Linked List (Recursive)

**When to use:** Recursive version of reversal.  
**LeetCode Example:** [Reverse Linked List - LeetCode #206](https://leetcode.com/problems/reverse-linked-list/)


```java
ListNode reverseList(ListNode head) {
    if (head == null || head.next == null) return head;
    ListNode newHead = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return newHead;
}
```

---

## 💥 4. Detect Cycle in Linked List (Floyd's Algorithm)

**When to use:** Detect loops in a linked list.  
**LeetCode Example:** [Linked List Cycle - LeetCode #141](https://leetcode.com/problems/linked-list-cycle/)


```java
boolean hasCycle(ListNode head) {
    ListNode slow = head, fast = head;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
        if (slow == fast) return true;
    }
    return false;
}
```

---

## 🧹 5. Remove N-th Node from End

**When to use:** Two-pointer trick.  
**LeetCode Example:** [Remove Nth Node From End of List - LeetCode #19](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)


```java
ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    ListNode first = dummy, second = dummy;
    for (int i = 0; i <= n; i++) first = first.next;
    while (first != null) {
        first = first.next;
        second = second.next;
    }
    second.next = second.next.next;
    return dummy.next;
}
```

---

## 🔀 6. Merge Two Sorted Lists

**When to
```java
ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(0);
    ListNode current = dummy;
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

## 🔁 7. Find Middle of Linked List
**When to use:** Slow and fast pointer trick.
**LeetCode Example:** [Middle of the Linked List - LeetCode #876](https://leetcode.com/problems/middle-of-the-linked-list/)

```java
ListNode middleNode(ListNode head) {
    ListNode slow = head, fast = head;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    return slow;
}
```

