
nums1 = input()
nums2 = input()

# 找到最长公共子数组
s = set()
n,m = len(nums1),len(nums2)

for i in range(n):
    for j in range(i+1,n+1):
        s.add(nums1[i:j+1])

res = ""

for i in range(m):
    for j in range(i+1,m+1):
        if nums2[i:j+1] in s and j+1-i > len(res):
            res = nums2[i:j+1]

if res:
    print(res.strip())
else:
    print(-1)