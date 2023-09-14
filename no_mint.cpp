#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <bits/stdc++.h>
#include <type_traits>
#define int long long 
#define PI 3.14159265359
using namespace std;

constexpr long long safe_mod(long long x, long long m) {
    x %= m;
    if (x < 0) x += m;
    return x;
}

pair<long long, long long> inv_gcd(long long a, long long b) {
    a = safe_mod(a, b);
    if (a == 0) return {b, 0};

    long long s = b, t = a;
    long long m0 = 0, m1 = 1;

    while (t) {
        long long u = s / t;
        s -= t * u;
        m0 -= m1 * u;  

        auto tmp = s;
        s = t;
        t = tmp;
        tmp = m0;
        m0 = m1;
        m1 = tmp;
    }
    if (m0 < 0) m0 += b / s;
    return {s, m0};
}

long long pow_mod(long long x, long long n, int m) {
    if (m == 1) return 0;
    unsigned int _m = (unsigned int)(m);
    unsigned long long r = 1;
    unsigned long long y = safe_mod(x, m);
    while (n) {
        if (n & 1) r = (r * y) % _m;
        y = (y * y) % _m;
        n >>= 1;
    }
    return r;
}

long long inv_mod(long long x, long long m) {
    assert(1 <= m);
    auto z = inv_gcd(x, m);
    assert(z.first == 1);
    return z.second;
}

using ll = long long;
using ld = long double;
using vi = vector<int>;
using vvi = vector<vi>;
using pii = pair<int,int>;
using vp = vector<pii>;
using vvp = vector<vp>;
using vs = vector<string>;
using vvs = vector<vs>;
using ti3 = tuple<int,int,int>;
using vti3 = vector<ti3>;
using ti4 = tuple<int,int,int,int>;
using vti4 = vector<ti4>;

template<typename T>
using min_pq = priority_queue<T, vector<T>, greater<T>>;
template<typename T>
using max_pq = priority_queue<T>;


template<int D, typename T>
struct Vec : public vector<Vec<D - 1, T>> {
  static_assert(D >= 1, "Vector dimension must be greater than zero!");
  template<typename... Args>
  Vec(int n = 0, Args... args) : vector<Vec<D - 1, T>>(n, Vec<D - 1, T>(args...)) {}
};

template<typename T>
struct Vec<1, T> : public vector<T> {
  Vec(int n = 0, const T& val = T()) : vector<T>(n, val) {}
};

template<typename T>
void printv(vector<T> v) {
    for (auto e : v) {
        cout << e << " ";
    }   cout << "\n";
}
 
template<typename T>
void printvv(vector<T> vv) {
    for (int i=0; i<vv.size(); i++) {
        cout << i << ": ";
        for (auto e : vv[i]) {
            cout << e << " ";
        }   cout << "\n";
    }
}

template<typename T>
void ri(T &x) {
    cin >> x;
}
template<typename T, typename... Args>
void ri(T &x, Args&... args) {
    ri(x);
    ri(args...) ;
}
template<typename T>
void ri(vector<T> &v) {
    for (auto &x : v) {
        cin >> x;
    }
}
template<typename T, typename... Args>
void ri(vector<T> &v, Args&... args) {
    ri(v);
    ri(args...);
}

template<typename T>
void po(T x) {
    cout << x << "\n";
}
template<typename T, typename... Args>
void po(T x, Args... args) {
    cout << x << " ";
    po(args...) ;
}
template<typename T>
void po(vector<T> &a) {
    int sz = a.size();
    for (int i=0; i<sz; i++) {
        cout << a[i] << ((i==sz-1)?"\n":" ");
    }
}

void __print(int x) {cerr << x;}
void __print(signed x) {cerr << x;}
void __print(long x) {cerr << x;}
void __print(unsigned x) {cerr << x;}
void __print(unsigned long x) {cerr << x;}
void __print(unsigned long long x) {cerr << x;}
void __print(float x) {cerr << x;}
void __print(double x) {cerr << x;}
void __print(long double x) {cerr << x;}
void __print(char x) {cerr << '\'' << x << '\'';}
void __print(const char *x) {cerr << '\"' << x << '\"';}
void __print(const string &x) {cerr << '\"' << x << '\"';}
void __print(bool x) {cerr << (x ? "true" : "false");}

template<typename T, typename V>
void __print(const pair<T, V> &x) {cerr << '{'; __print(x.first); cerr << ','; __print(x.second); cerr << '}';}
template<typename T1, typename T2, typename T3>
void __print(const tuple<T1, T2, T3> &x) {cerr << '{'; __print(get<0>(x)); cerr << ','; __print(get<1>(x)); cerr << ','; __print(get<2>(x)); cerr << '}';}
template<typename T1, typename T2, typename T3, typename T4>
void __print(const tuple<T1, T2, T3, T4> &x) {cerr << '{'; __print(get<0>(x)); cerr << ','; __print(get<1>(x)); cerr << ','; __print(get<2>(x)); cerr << ','; __print(get<3>(x)); cerr << '}';}
template<typename T>
void __print(const T &x) {int f = 0; cerr << '{'; for (auto &i: x) cerr << (f++ ? "," : ""), __print(i); cerr << "}";}
template<typename T1, typename T2>
void __print(map<T1,T2> &mp) {for (auto [k,v] : mp) {cerr << '{'; __print(k); cerr << ':'; __print(v); cerr << '}';}}
void _print() {cerr << "]\n";}
template <typename T, typename... V>
void _print(T t, V... v) {__print(t); if (sizeof...(v)) cerr << ", "; _print(v...);}
#ifndef ONLINE_JUDGE
#define debug(x...) cerr << "[" << #x << "] = ["; _print(x)
#else
#define debug(x...)
#endif

int cnt_leq_x(vi &a, int x) {
    return upper_bound(a.begin(), a.end(), x) - a.begin();
}

int cnt_leq_x(vi &a, int x, int lo, int hi) {
    return upper_bound(a.begin()+lo, a.begin()+hi, x) - a.begin()+lo;
}

int cnt_lt_x(vi &a, int x) {
    return lower_bound(a.begin(), a.end(), x) - a.begin();
}

int cnt_lt_x(vi &a, int x, int lo, int hi) {
    return lower_bound(a.begin()+lo, a.begin()+hi, x) - a.begin()+lo;
}

int cnt_geq_x(vi &a, int x) {
    return a.end() - lower_bound(a.begin(), a.end(), x);
}

int cnt_geq_x(vi &a, int x, int lo, int hi) {
    return a.begin()+hi - lower_bound(a.begin()+lo, a.begin()+hi, x);
}

int cnt_gt_x(vi &a, int x) {
    return a.end() - upper_bound(a.begin(), a.end(), x);
}

int cnt_gt_x(vi &a, int x, int lo, int hi) {
    return a.begin()+hi - upper_bound(a.begin()+lo, a.begin()+hi, x);
}

bool mul_overflow(int a, int b) {
    int c;
    return __builtin_mul_overflow(a, b, &c);
}

template<typename T>
int popcount(T x) {return __builtin_popcount(x);}

template<typename T>
T sum(vector<T> &a) {
    T ret = 0;
    for (auto v : a) ret += v;
    return ret;
}

template<typename T>
T max(vector<T> &a) {
    return *max_element(a.begin(), a.end());
}

template<typename T>
T min(vector<T> &a) {
    return *min_element(a.begin(), a.end());
}

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

int int_pow(int base, int exp) {
    int res = 1;
    while (exp) {
        if (exp & 1) res *= base;
        exp >>= 1;
        base *= base;
    }
    return res;
}

int highest_power_of_2(int n) {
    while((n & (n-1)) != 0){
        n = n & (n-1);
    }
    return n;
}

int msb_pos(int x) {
    if (x==0) return -1;
    int y = __builtin_clzll(x);
    int ret = 63 - y;
    return ret;
}

template<typename T>
void chmax(T &x, T y) {x = max(x,y);}

template<typename T>
void chmin(T &x, T y) {x = min(x,y);}

template<typename T>
void asort(vector<T> &a) {sort(a.begin(), a.end());}

template<typename T>
void dsort(vector<T> &a) {sort(a.rbegin(), a.rend());}

template<typename T>
void reverse(vector<T> &a) {reverse(a.begin(), a.end());}

template<typename T>
set<T> get_set(vector<T> &a) {
    set<T> ret(a.begin(), a.end());
    return ret;
}

template<typename T>
vector<T> get_unique(vector<T> a) {
    asort(a);
    a.erase(unique(a.begin(), a.end()), a.end());
    return a;
}

int ccw(pii p1, pii p2, pii p3) {
    auto [x1,y1] = p1;
    auto [x2,y2] = p2;
    auto [x3,y3] = p3;
    return (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1);
}

pii extgcd(int a, int b) {
    if (b==0) return {1,0};
    int q = a / b;
    auto [x,y] = extgcd(b,a-b*q);
    return {y,x-q*y};
}

vector<string> split_str(string s, const char delim = ' ') {
    vector<string> ret;
    stringstream ss(s);
    string t;
    while (getline(ss, t, delim)) {
        ret.push_back(t);
    }
    return ret;
}

struct dsu {
  public:
    dsu() : _n(0) {}
    dsu(int n) : _n(n), parent_or_size(n, -1) {}

    int merge(int a, int b) {
        assert(0 <= a && a < _n);
        assert(0 <= b && b < _n);
        int x = leader(a), y = leader(b);
        if (x == y) return x;
        if (-parent_or_size[x] < -parent_or_size[y]) swap(x, y);
        parent_or_size[x] += parent_or_size[y];
        parent_or_size[y] = x;
        return x;
    }

    bool same(int a, int b) {
        assert(0 <= a && a < _n);
        assert(0 <= b && b < _n);
        return leader(a) == leader(b);
    }

    int leader(int a) {
        assert(0 <= a && a < _n);
        if (parent_or_size[a] < 0) return a;
        return parent_or_size[a] = leader(parent_or_size[a]);
    }

    int size(int a) {
        assert(0 <= a && a < _n);
        return -parent_or_size[leader(a)];
    }

    vector<vector<int>> groups() {
        vector<int> leader_buf(_n), group_size(_n);
        for (int i = 0; i < _n; i++) {
            leader_buf[i] = leader(i);
            group_size[leader_buf[i]]++;
        }
        vector<vector<int>> result(_n);
        for (int i = 0; i < _n; i++) {
            result[i].reserve(group_size[i]);
        }
        for (int i = 0; i < _n; i++) {
            result[leader_buf[i]].push_back(i);
        }
        result.erase(
            remove_if(result.begin(), result.end(),
                [&](const vector<int>& v) { return v.empty(); }),
            result.end());
        return result;
    }

  private:
    int _n;
    vector<int> parent_or_size;
};

class Trie {
    public:

    bool leaf;
    Trie* children[26];

    Trie() {
        this->leaf = false;
        for (int i=0; i<26; i++) {
            this->children[i] = nullptr;
        }
    }

    void insert(string s) {
        Trie* node = this;

        for (int i=0; i<(int)s.size(); i++) {
            int idx = s[i] - 'a';
            if (node->children[idx] == nullptr) node->children[idx] = new Trie();
            node = node->children[idx];
        }
        node->leaf = true;
    }

    bool search(string key) {
        Trie* node = this;
        for (int i = 0; i <(int)key.size(); i++) {
            int idx = key[i] - 'a';
            if (!node->children[idx]) return false;
            node = node->children[idx];
        }
        return (node->leaf);
    }
};

////////////////////////////////////

vi primes;
vi spf; 
void init_spf(int n) {
    spf.resize(n+1);
    for (int i=2; i <= n; i++) {
        if (spf[i] == 0) {
            spf[i] = i;
            primes.push_back(i);
        }
        for (int j = 0; i * primes[j] <= n; j++) {
            spf[i * primes[j]] = primes[j];
            if (primes[j] == spf[i]) {
                break;
            }
        }
    }
}

vi get_pfactors(int x) {
    vector<int> ret;
    while (x != 1) {
        ret.push_back(spf[x]);
        x = x / spf[x];
    }
    return ret;
}

////////////////////////////////////

struct rabin_karp {
    int n,p,m;
    vi p_pow,p_inv,h;

    rabin_karp(string s, int p, int m) {
        this->n = s.size();
        this->p = p;
        this->m = m;
        p_pow = p_inv = vi(n+1,1);
        for (int i=1; i<=n; i++) p_pow[i] = (p_pow[i-1]*p)%m;
        p_inv[n] = pow_mod(p_pow[n],m-2,m);
        for (int i=n-1; i>0; i--) p_inv[i] = (p_inv[i+1]*p)%m;
        h = vi(n+1);
        for (int i=0; i<n; i++) {
            int x = ('a'<=s[i] && s[i]<='z') ? (s[i]-'a'+1) : (s[i]-'A'+27);
            h[i+1] = (h[i]+(x*p_pow[i]))%m;
        }
    }

    int query(int l, int r) {
        assert(0<=l && l<=r && r<n);
        int x = ((h[r+1]-h[l])%m+m)%m;
        return (x*p_inv[l]) % m;
    }
};

////////////////////////////////////

template<typename T> T op_max(T x, T y) {return max(x,y);}
template<typename T> T op_min(T x, T y) {return min(x,y);}

template<typename T,  T (*op)(T, T)>
struct sparse_table {
    int n,m;
    vector<vector<T>> table;

    inline T merge(T x, T y) {
        return op(x, y);
    }

    sparse_table(vector<T> &a) {
        n = a.size();
        m = __lg(n) + 1;
        table.assign(m, vector<T>(n));
        for (int i = 0; i < n; i++) table[0][i] = a[i];
        for (int i = 1; i < m; i++) {
            for (int j = 0; j + (1<<i) <= n; j++) {
                table[i][j] = merge(table[i-1][j], table[i-1][j + (1<<(i-1))]);
            }
        }
    }

    T query(int l, int r) {
        // l, r : inclusive
        assert(l<=r && 0<=l && r< n);
        int u = __lg(r-l+1);
        return merge(table[u][l], table[u][r-(1<<u)+1]);
    }

    T query(int l, int r, T e) {
        // e for identity
        l = max(l,0ll);
        r = min(r,n-1);
        int u = __lg(r-l+1);
        if (l<=r) return merge(table[u][l], table[u][r-(1<<u)+1]);
        else return e;
    }
};

template<typename T> using max_spt = sparse_table<T,op_max>;
template<typename T> using min_spt = sparse_table<T,op_min>;

////////////////////////////////////

struct LCA {
    vi height, euler, pw2, lg2, idx;
    vvp sptable;
    int n, logn;

    LCA(vector<vector<int>> &adj, int root = 0) {
        n = adj.size();
        logn = ceil(log2(n))+1;
        height.resize(n);
        euler.reserve(2*n);
        idx.resize(n);
        sptable.assign(logn, vp(2*n));
        pw2.assign(logn, 1);
        for (int k=1; k<logn; k++) pw2[k] = 2*pw2[k-1];
        lg2.assign(2*n, -1);
        for(int k=0; k<logn; k++) {
            if(pw2[k] < 2*n) lg2[pw2[k]] = k;
        }
        for(int i=1; i<2*n; i++) {
            if(lg2[i] == -1) lg2[i] = lg2[i-1];
        }

        dfs(adj, root, -1);
        int m = euler.size();
        
        for(int i=0; i<m; i++) {
            sptable[0][i] = {height[euler[i]], euler[i]};
        }
        for(int k=1; k<logn; k++){
            for(int i=0; i<m; i++){
                if(i+pw2[k-1] > m) continue;
                sptable[k][i] = min(sptable[k-1][i], sptable[k-1][i+pw2[k-1]]);
            }
        }    
    }

    void dfs(vector<vector<int>> &adj, int u, int p, int h = 0) {
        height[u] = h;
        idx[u] = euler.size();
        euler.push_back(u);
        for (auto v : adj[u]) {
            if (v == p) continue;
            dfs(adj, v, u, h + 1);
            euler.push_back(u);
        }
    }

    int query(int u, int v) {
        int l = idx[u], r = idx[v];
        if(l > r) swap(l,r);
        int k = lg2[r-l+1];
        return min(sptable[k][l], sptable[k][r-pw2[k]+1]).second;
    }
};

template <class T> struct fenwick_tree {
    int n;
    vector<T> a;

    fenwick_tree(int n) {
        this->n = n;
        a.resize(n);
    }

    void add(int p, T x) {
        assert(0 <= p && p < n);
        p++;
        while (p <= n) {
            a[p - 1] += x;
            p += p & -p;
        }
    }

    T sum(int r) {
        T s = 0;
        while (r > 0) {
            s += a[r - 1];
            r -= r & -r;
        }
        return s;
    }

    T sum(int l, int r) {
        assert(0 <= l && l <= r && r <= n);
        return sum(r) - sum(l);
    }
};

struct segtree {
    ll n, sz, log=0;
    vi a;

    segtree(int n) {
        this->n = n;
        while ((1 << log) < n) log++;
        sz = (1 << log);
        a = vi(2*sz, e());
    }

    ll e() {return 0;}
    ll op(ll x, ll y) {return x+y;}
    void update(int p) {a[p] = op(a[2*p],a[2*p+1]);}

    void set(int p, ll x) {
        p += sz;
        a[p] = x;
        while (p>>=1) update(p);
    }

    ll prod(int l, int r) {
        l += sz, r += sz;
        ll lp = e(), rp = e();
        while (l < r) {
            if (l&1) lp = op(lp, a[l++]);
            if (r&1) rp = op(rp, a[--r]);
            l >>= 1; r >>= 1;
        }
        return op(lp,rp);
    }

    ll get(int p) {return a[p+sz];}
};


////////////////////////////////////

// <lazy segtree prototype>

// using S = pii;
// using F = pii;

// S op(S a, S b) {
//     auto [x,u] = a;
//     auto [y,v] = b;
//     return {x+y,u+v};
// }
 
// S e() {
//     return {0,0};
// }
 
// S mapping(F f, S s) {
//     auto [a,b] = f;
//     auto [x,y] = s;
//     return {a*x + b*y, y};
// }
 
// F composition(F f, F g) {
//     auto [a,b] = g;
//     auto [c,d] = f;
//     return {c*a, c*b+d};
// }

// F id () {
//     return {1,0};
// }
 
// lazy_segtree<S, op, e, F, mapping, composition, id> seg(n);

////////////////////////////////////

void io_util() {
    #ifdef LOCAL
    freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    #endif
    cin.tie(0)->sync_with_stdio(0);
    cout.precision(17);
}

////////////////////////////////////

void solve(); 
signed main() {
    io_util();
    int tc = 1;
    // ri(tc);
    for (int i=1; i<=tc; i++) {
        // cout << "Case #" << i << ": ";
        solve();
    }
    return 0;
}

void solve() {
}