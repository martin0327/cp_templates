#include <bits/stdc++.h>
using namespace std;

// struct item {
//   int sum;
// };
typedef int item;
 
struct segtree{
  int size;
  vector<item> arr;
 
  const item neutral_item = 0;
 
  item single (int v) {
    return v;
  }
 
  item merge(item a, item b) {
    return max(a,b);
  }
 
  void init(int n) {
    size = 1;
    while (size < n) size *= 2;
    arr.resize(2*size);
  }
 
  void build(vi &a, int x, int lx, int rx) {
    if (rx - lx == 1) {
      if (0<=lx && lx < a.size()) {
        arr[x] = single(a[lx]);
      }
      return;
    }
    int m = (lx + rx) / 2;
    build(a, 2*x+1, lx, m);
    build(a, 2*x+2, m, rx);
 
    arr[x] = merge(arr[2*x+1], arr[2*x+2]);
  }
 
  void build(vi &a) {
    build(a, 0, 0, size);
  }
 
  void set(int i, int v, int x, int lx, int rx) {
    if (rx - lx == 1) {
      arr[x] = single(v);
      return;
    }
 
    int m = (lx + rx) / 2;
 
    if (i < m) set(i, v, 2*x+1, lx, m);
    else set(i, v, 2*x+2, m, rx);
 
    arr[x] = merge(arr[2*x+1], arr[2*x+2]);
  }
 
  void set(int i, int v) {
    set(i, v, 0, 0, size);
  }
 
  item calc(int l, int r, int x, int lx, int rx) {
    if (r<=lx || rx <= l) return neutral_item; 
    if (l<=lx && rx <= r) return arr[x];
 
    int m = (lx+rx) / 2;
    
    item left = calc(l, r, 2*x+1, lx, m);
    item right = calc(l, r, 2*x+2, m, rx);
 
    return merge(left, right);
  }
 
  item calc(int l, int r) {
    return calc(l, r, 0, 0, size);
  }
 
  int find(int k, int l, int x, int lx, int rx) {
    if (rx - lx == 1) {
      if (0<=lx && lx<=size) {
        if (arr[x] >= k && lx >=l) return lx;
        else return -1;
      }
    }
    int m = (lx + rx) / 2;
    int left = arr[2*x+1];
 
    if (l<m && k<=left) {
      int temp = find(k,l,2*x+1,lx,m);
      if (temp == -1) return find(k,l,2*x+2,m,rx);
      else return temp;
    } else {
      return find(k,l,2*x+2,m,rx);
    }
  }
 
  int find(int k, int l) {
    return find(k,l,0,0,size);
  }
};
