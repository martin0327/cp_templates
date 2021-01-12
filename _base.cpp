#include <bits/stdc++.h>
#define endl "\n"
#define int long long
  
using namespace std;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef pair<int,int> pii;

template<typename T>
void printv(vector<T> v) {
  for (auto e : v) {
    cout << e << " ";
  } cout << endl;
}
 
template<typename T>
void printvv(vector<T> vv) {
  for (int i=0; i<vv.size(); i++) {
    cout << i << ": ";
    for (auto e : vv[i]) {
      cout << e << " ";
    } cout << endl;
  }
}

///////////////////////////////////////////////////////////////////////


signed main() {
  #ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  // freopen("output.txt", "w", stdout);
  #endif
  ios_base::sync_with_stdio(false); 
  cin.tie(nullptr); 
  cout.precision(17);

  
  return 0;
}