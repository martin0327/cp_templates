#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <bits/stdc++.h>
#include <type_traits>
#define int long long 
#define PI 3.14159265359
using namespace std;

constexpr int bsf_constexpr(unsigned int n) {
    int x = 0;
    while (!(n & (1 << x))) x++;
    return x;
}

int bsf(unsigned int n) {
    return __builtin_ctz(n);
}

constexpr long long safe_mod(long long x, long long m) {
    x %= m;
    if (x < 0) x += m;
    return x;
}

constexpr long long pow_mod_constexpr(long long x, long long n, int m) {
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

constexpr bool is_prime_constexpr(int n) {
    if (n <= 1) return false;
    if (n == 2 || n == 7 || n == 61) return true;
    if (n % 2 == 0) return false;
    long long d = n - 1;
    while (d % 2 == 0) d /= 2;
    constexpr long long bases[3] = {2, 7, 61};
    for (long long a : bases) {
        long long t = d;
        long long y = pow_mod_constexpr(a, t, n);
        while (t != n - 1 && y != 1 && y != n - 1) {
            y = y * y % n;
            t <<= 1;
        }
        if (y != n - 1 && t % 2 == 0) {
            return false;
        }
    }
    return true;
}
template <int n> constexpr bool is_prime = is_prime_constexpr(n);

pair<long long, long long> inv_gcd(long long a, long long b) {
    a = safe_mod(a, b);
    if (a == 0) return {b, 0};

    long long s = b, t = a;
    long long m0 = 0, m1 = 1;

    while (t) {
        long long u = s / t;
        s -= t * u;
        m0 -= m1 * u;  // |m1 * u| <= |m1| * s <= b

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

template <int m>
struct static_modint {
    using mint = static_modint;

  public:
    static constexpr int mod() { return m; }
    static mint raw(int v) {
        mint x;
        x._v = v;
        return x;
    }

    static_modint() : _v(0) {}
    template <class T>
    static_modint(T v) {
        long long x = (long long)(v % (long long)(umod()));
        if (x < 0) x += umod();
        _v = (unsigned int)(x);
    }

    unsigned int val() const { return _v; }

    mint& operator++() {
        _v++;
        if (_v == umod()) _v = 0;
        return *this;
    }
    mint& operator--() {
        if (_v == 0) _v = umod();
        _v--;
        return *this;
    }
    mint operator++(signed) {
        mint result = *this;
        ++*this;
        return result;
    }
    mint operator--(signed) {
        mint result = *this;
        --*this;
        return result;
    }

    mint& operator+=(const mint& rhs) {
        _v += rhs._v;
        if (_v >= umod()) _v -= umod();
        return *this;
    }
    mint& operator-=(const mint& rhs) {
        _v -= rhs._v;
        if (_v >= umod()) _v += umod();
        return *this;
    }
    mint& operator*=(const mint& rhs) {
        unsigned long long z = _v;
        z *= rhs._v;
        _v = (unsigned int)(z % umod());
        return *this;
    }
    mint& operator/=(const mint& rhs) { return *this = *this * rhs.inv(); }

    mint operator+() const { return *this; }
    mint operator-() const { return mint() - *this; }

    mint pow(long long n) const {
        assert(0 <= n);
        mint x = *this, r = 1;
        while (n) {
            if (n & 1) r *= x;
            x *= x;
            n >>= 1;
        }
        return r;
    }
    mint inv() const {
        if (prime) {
            assert(_v);
            return pow(umod() - 2);
        } else {
            auto eg = inv_gcd(_v, m);
            assert(eg.first == 1);
            return eg.second;
        }
    }

    friend mint operator+(const mint& lhs, const mint& rhs) {
        return mint(lhs) += rhs;
    }
    friend mint operator-(const mint& lhs, const mint& rhs) {
        return mint(lhs) -= rhs;
    }
    friend mint operator*(const mint& lhs, const mint& rhs) {
        return mint(lhs) *= rhs;
    }
    friend mint operator/(const mint& lhs, const mint& rhs) {
        return mint(lhs) /= rhs;
    }
    friend bool operator==(const mint& lhs, const mint& rhs) {
        return lhs._v == rhs._v;
    }
    friend bool operator!=(const mint& lhs, const mint& rhs) {
        return lhs._v != rhs._v;
    }

  private:
    unsigned int _v;
    static constexpr unsigned int umod() { return m; }
    static constexpr bool prime = is_prime<m>;
};

int ceil_pow2(int n) {
    int x = 0;
    while ((1U << x) < n) x++;
    return x;
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

pair<long long, long long> crt(const vector<long long>& r,
                                    const vector<long long>& m) {
    assert(r.size() == m.size());
    int n = r.size();
    long long r0 = 0, m0 = 1;
    for (int i = 0; i < n; i++) {
        assert(1 <= m[i]);
        long long r1 = safe_mod(r[i], m[i]), m1 = m[i];
        if (m0 < m1) {
            swap(r0, r1);
            swap(m0, m1);
        }
        if (m0 % m1 == 0) {
            if (r0 % m1 != r1) return {0, 0};
            continue;
        }


        long long g, im;
        tie(g, im) = inv_gcd(m0, m1);

        long long u1 = (m1 / g);
        if ((r1 - r0) % g) return {0, 0};

        long long x = (r1 - r0) / g % u1 * im % u1;

        r0 += x * m0;
        m0 *= u1;  // -> lcm(m0, m1)
        if (r0 < 0) r0 += m0;
    }
    return {r0, m0};
}

using modint998244353 = static_modint<998244353>;
using modint1000000007 = static_modint<1000000007>;
using mint = modint998244353;
// using mint = modint1000000007;
using ld = long double;
using vi = vector<int>;
using vvi = vector<vi>;
using vm = vector<mint>;
using vvm = vector<vm>;
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
void po(mint x) {
    cout << x.val() << "\n";
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
void po(vector<mint> &a) {
    int sz = a.size();
    for (int i=0; i<sz; i++) {
        cout << a[i].val() << ((i==sz-1)?"\n":" ");
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
void __print(mint x) {cerr << x.val();}

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
    Trie* ch[26];

    Trie() {
        this->leaf = false;
        for (int i=0; i<26; i++) {
            this->ch[i] = nullptr;
        }
    }

    void insert(string s) {
        Trie* node = this;

        for (int i=0; i<(int)s.size(); i++) {
            int idx = s[i] - 'a';
            if (node->ch[idx] == nullptr) node->ch[idx] = new Trie();
            node = node->ch[idx];
        }
        node->leaf = true;
    }

    bool search(string key) {
        Trie* node = this;
        for (int i = 0; i <(int)key.size(); i++) {
            int idx = key[i] - 'a';
            if (!node->ch[idx]) return false;
            node = node->ch[idx];
        }
        return (node->leaf);
    }
};

////////////////////////////////////
 
vector<mint> fact;
vector<mint> finv;
 
void init_fact(int fact_sz, int finv_sz) {
    assert(fact_sz >= finv_sz);
    fact.resize(fact_sz+1,1);
    finv.resize(finv_sz+1);
    for (int i=1; i<=fact_sz; i++) {
        fact[i] = fact[i-1] * i;
    }
    finv[finv_sz] = fact[finv_sz].inv();
    for (int i=finv_sz; i>0; i--) {
        finv[i-1] = finv[i] * i;
    }
}
 
mint ncr(int n, int r) {
    mint numer = fact[n];
    mint denom = finv[r] * finv[n-r];
    return numer * denom;
}

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

template <class S, S (*op)(S, S), S (*e)()> struct segtree {
  public:
    segtree() : segtree(0) {}
    explicit segtree(int n) : segtree(vector<S>(n, e())) {}
    explicit segtree(const vector<S>& v) : _n((int)v.size()) {
        log = ceil_pow2(_n);
        size = 1 << log;
        d = vector<S>(2 * size, e());
        for (int i = 0; i < _n; i++) d[size + i] = v[i];
        for (int i = size - 1; i >= 1; i--) {
            update(i);
        }
    }

    void set(int p, S x) {
        assert(0 <= p && p < _n);
        p += size;
        d[p] = x;
        for (int i = 1; i <= log; i++) update(p >> i);
    }

    S get(int p) const {
        assert(0 <= p && p < _n);
        return d[p + size];
    }

    S prod(int l, int r) const {
        assert(0 <= l && l <= r && r <= _n);
        S sml = e(), smr = e();
        l += size;
        r += size;

        while (l < r) {
            if (l & 1) sml = op(sml, d[l++]);
            if (r & 1) smr = op(d[--r], smr);
            l >>= 1;
            r >>= 1;
        }
        return op(sml, smr);
    }

    S all_prod() const { return d[1]; }

    template <bool (*f)(S)> int max_right(int l) const {
        return max_right(l, [](S x) { return f(x); });
    }
    template <class F> int max_right(int l, F f) const {
        assert(0 <= l && l <= _n);
        assert(f(e()));
        if (l == _n) return _n;
        l += size;
        S sm = e();
        do {
            while (l % 2 == 0) l >>= 1;
            if (!f(op(sm, d[l]))) {
                while (l < size) {
                    l = (2 * l);
                    if (f(op(sm, d[l]))) {
                        sm = op(sm, d[l]);
                        l++;
                    }
                }
                return l - size;
            }
            sm = op(sm, d[l]);
            l++;
        } while ((l & -l) != l);
        return _n;
    }

    template <bool (*f)(S)> int min_left(int r) const {
        return min_left(r, [](S x) { return f(x); });
    }
    template <class F> int min_left(int r, F f) const {
        assert(0 <= r && r <= _n);
        assert(f(e()));
        if (r == 0) return 0;
        r += size;
        S sm = e();
        do {
            r--;
            while (r > 1 && (r % 2)) r >>= 1;
            if (!f(op(d[r], sm))) {
                while (r < size) {
                    r = (2 * r + 1);
                    if (f(op(d[r], sm))) {
                        sm = op(d[r], sm);
                        r--;
                    }
                }
                return r + 1 - size;
            }
            sm = op(d[r], sm);
        } while ((r & -r) != r);
        return 0;
    }

  private:
    int _n, size, log;
    vector<S> d;

    void update(int k) { d[k] = op(d[2 * k], d[2 * k + 1]); }
};

template <class S,
          S (*op)(S, S),
          S (*e)(),
          class F,
          S (*mapping)(F, S),
          F (*composition)(F, F),
          F (*id)()>
struct lazy_segtree {
  public:
    lazy_segtree() : lazy_segtree(0) {}
    explicit lazy_segtree(int n) : lazy_segtree(vector<S>(n, e())) {}
    explicit lazy_segtree(const vector<S>& v) : _n((int)(v.size())) {
        log = ceil_pow2(_n);
        size = 1 << log;
        d = vector<S>(2 * size, e());
        lz = vector<F>(size, id());
        for (int i = 0; i < _n; i++) d[size + i] = v[i];
        for (int i = size - 1; i >= 1; i--) {
            update(i);
        }
    }

    void set(int p, S x) {
        assert(0 <= p && p < _n);
        p += size;
        for (int i = log; i >= 1; i--) push(p >> i);
        d[p] = x;
        for (int i = 1; i <= log; i++) update(p >> i);
    }

    S get(int p) {
        assert(0 <= p && p < _n);
        p += size;
        for (int i = log; i >= 1; i--) push(p >> i);
        return d[p];
    }

    S prod(int l, int r) {
        assert(0 <= l && l <= r && r <= _n);
        if (l == r) return e();

        l += size;
        r += size;

        for (int i = log; i >= 1; i--) {
            if (((l >> i) << i) != l) push(l >> i);
            if (((r >> i) << i) != r) push((r - 1) >> i);
        }

        S sml = e(), smr = e();
        while (l < r) {
            if (l & 1) sml = op(sml, d[l++]);
            if (r & 1) smr = op(d[--r], smr);
            l >>= 1;
            r >>= 1;
        }

        return op(sml, smr);
    }

    S all_prod() { return d[1]; }

    void apply(int p, F f) {
        assert(0 <= p && p < _n);
        p += size;
        for (int i = log; i >= 1; i--) push(p >> i);
        d[p] = mapping(f, d[p]);
        for (int i = 1; i <= log; i++) update(p >> i);
    }
    void apply(int l, int r, F f) {
        assert(0 <= l && l <= r && r <= _n);
        if (l == r) return;

        l += size;
        r += size;

        for (int i = log; i >= 1; i--) {
            if (((l >> i) << i) != l) push(l >> i);
            if (((r >> i) << i) != r) push((r - 1) >> i);
        }

        {
            int l2 = l, r2 = r;
            while (l < r) {
                if (l & 1) all_apply(l++, f);
                if (r & 1) all_apply(--r, f);
                l >>= 1;
                r >>= 1;
            }
            l = l2;
            r = r2;
        }

        for (int i = 1; i <= log; i++) {
            if (((l >> i) << i) != l) update(l >> i);
            if (((r >> i) << i) != r) update((r - 1) >> i);
        }
    }

    template <bool (*g)(S)> int max_right(int l) {
        return max_right(l, [](S x) { return g(x); });
    }
    template <class G> int max_right(int l, G g) {
        assert(0 <= l && l <= _n);
        assert(g(e()));
        if (l == _n) return _n;
        l += size;
        for (int i = log; i >= 1; i--) push(l >> i);
        S sm = e();
        do {
            while (l % 2 == 0) l >>= 1;
            if (!g(op(sm, d[l]))) {
                while (l < size) {
                    push(l);
                    l = (2 * l);
                    if (g(op(sm, d[l]))) {
                        sm = op(sm, d[l]);
                        l++;
                    }
                }
                return l - size;
            }
            sm = op(sm, d[l]);
            l++;
        } while ((l & -l) != l);
        return _n;
    }

    template <bool (*g)(S)> int min_left(int r) {
        return min_left(r, [](S x) { return g(x); });
    }
    template <class G> int min_left(int r, G g) {
        assert(0 <= r && r <= _n);
        assert(g(e()));
        if (r == 0) return 0;
        r += size;
        for (int i = log; i >= 1; i--) push((r - 1) >> i);
        S sm = e();
        do {
            r--;
            while (r > 1 && (r % 2)) r >>= 1;
            if (!g(op(d[r], sm))) {
                while (r < size) {
                    push(r);
                    r = (2 * r + 1);
                    if (g(op(d[r], sm))) {
                        sm = op(d[r], sm);
                        r--;
                    }
                }
                return r + 1 - size;
            }
            sm = op(d[r], sm);
        } while ((r & -r) != r);
        return 0;
    }

  private:
    int _n, size, log;
    vector<S> d;
    vector<F> lz;

    void update(int k) { d[k] = op(d[2 * k], d[2 * k + 1]); }
    void all_apply(int k, F f) {
        d[k] = mapping(f, d[k]);
        if (k < size) lz[k] = composition(f, lz[k]);
    }
    void push(int k) {
        all_apply(2 * k, lz[k]);
        all_apply(2 * k + 1, lz[k]);
        lz[k] = id();
    }
};

template <class T> vector<int> z_algorithm(const vector<T>& s) {
    int n = (int)(s.size());
    if (n == 0) return {};
    vector<int> z(n);
    z[0] = 0;
    for (int i = 1, j = 0; i < n; i++) {
        int& k = z[i];
        k = (j + z[j] <= i) ? 0 : min(j + z[j] - i, z[i - j]);
        while (i + k < n && s[k] == s[i + k]) k++;
        if (j + z[j] < i + z[i]) j = i;
    }
    z[0] = n;
    return z;
}

vector<int> z_algorithm(const string& s) {
    int n = s.size();
    vector<int> s2(n);
    for (int i = 0; i < n; i++) {
        s2[i] = s[i];
    }
    return z_algorithm(s2);
}

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