#include <type_traits>
#include <utility>

struct nonesuch
{
    ~nonesuch() = delete;
    nonesuch(nonesuch const&) = delete;
    void operator=(nonesuch const&) = delete;
};

template <typename T>
struct type_identity
{
    using type = T;
};

template<char... Es>
struct literal_list {};
template<typename... Ts>
struct type_list {};

namespace mpl {
    template<typename T> struct lazy { using type = typename T::type; };

    template<bool Cond, typename True, typename False>
    struct If
    {
        using type = False;
    };
    template<typename True, typename _>
    struct If<true, True, _>
    {
        using type = True;
    };

    template<char C, template<char> class... Cases>
    struct Switch;

    namespace impl {
        template<bool Cond, char C, template<char> class... Cases>
        struct Switch
        {
            static_assert(sizeof...(Cases) > 0, "No case matches value");
        };

        template<char C, template<char> class Case, template<char> class... Cases>
        struct Switch<true, C, Case, Cases...>
        {
            using type = Case<C>;
        };
        template<char C, template<char> class Case, template<char> class... Cases>
        struct Switch<false, C, Case, Cases...>
        {
            using type = typename mpl::Switch<C, Cases...>::type;
        };
    }

    template<char C, template<char> class... Cases>
    struct Switch
    {
        static_assert(sizeof...(Cases) > 0, "No case matches value");
    };

    template<char C, template<char> class Case, template<char> class... Cases>
    struct Switch<C, Case, Cases...>
    {
        using type = typename impl::Switch<Case<C>::cond_v, C, Case, Cases...>::type;
    };

    template<typename L, template<auto> class Op>
    struct list_map_from_literal;
    template<char... Cs, template<auto> class Op>
    struct list_map_from_literal<literal_list<Cs...>, Op>
    {
        using type = type_list<typename Op<Cs>::type...>;
    };

    template<typename L, template<typename> class Op>
    struct list_find;
    template<typename L, typename... Ls, template<typename> class Op>
    struct list_find<type_list<L, Ls...>, Op>
    {
        using res_type = typename mpl::If<Op<L>::cond_v, type_identity<L>, list_find<type_list<Ls...>, Op>>::type;
        using type = typename res_type::type;
    };

    template<typename L, typename InitAcc, template<typename Acc, typename Elem> class Op>
    struct list_fold;
    template<typename InitAcc, template<typename,typename> class Op>
    struct list_fold<type_list<>, InitAcc, Op>
    {
        using type = InitAcc;
    };
    template<typename L, typename... Ls, typename InitAcc, template<typename, typename> class Op>
    struct list_fold<type_list<L, Ls...>, InitAcc, Op>
    {
        using type = typename list_fold<type_list<Ls...>, typename Op<InitAcc, L>::type, Op>::type;
    };
}

namespace impl {
    template <const char* Str, typename T>
    struct string_to_charlist;

    template <const char* Str, std::size_t... Indices>
    struct string_to_charlist<Str, std::index_sequence<Indices...>>
    {
        using type = literal_list<Str[Indices]...>;
    };
}

template<typename T, std::decay_t<T> V>
struct string_to_charlist;
template<int N, const char* Str>
struct string_to_charlist<const char[N], Str>
{
    using type = typename impl::string_to_charlist<Str, std::make_index_sequence<N - 1>>::type;
};

static constexpr const char empty[] = "";
static constexpr const char hello[] = "hello";
static_assert(std::is_same<string_to_charlist<decltype(hello), hello>::type, literal_list<'h', 'e', 'l', 'l', 'o'>>::value);
static_assert(std::is_same<string_to_charlist<decltype(empty), empty>::type, literal_list<>>::value);

template <char C> struct letter_kind;
template <int I> struct num_kind;
struct blank_kind;
struct open_paren;
struct close_paren;

template <char C>
struct whitespace_case
{
    static constexpr bool cond_v = (C == ' ' || C == '\t' || C == '\r' || C == '\n');
    using type = blank_kind;
};
template <char C>
struct num_case
{
    static constexpr bool cond_v = (C >= '0' && C <= '9');
    using type = num_kind<C - '0'>;
};
template <char C>
struct letter_case
{
    static constexpr bool cond_v = !(C >= '0' && C <= '9') && !(C == ' ' || C == '\t' || C == '\r' || C == '\n') && C != '(' && C != ')';
    using type = letter_kind<C>;
};
template<char C>
struct open_paren_case
{
    static constexpr bool cond_v = (C == '(');
    using type = open_paren;
};
template<char C>
struct close_paren_case
{
    static constexpr bool cond_v = (C == ')');
    using type = close_paren;
};


template<char C> struct box
{
    using result_case = typename mpl::Switch<C, open_paren_case, close_paren_case, whitespace_case, num_case, letter_case>::type;
    using type = typename result_case::type;
};

static_assert(std::is_same_v<typename mpl::list_map_from_literal<literal_list<' ', 'c', '1', '+'>, box>::type, type_list<blank_kind, letter_kind<'c'>, num_kind<1>, letter_kind<'+'>>>);

template<char...> struct lexem_ident;
template<int> struct lexem_int;
template<bool> struct lexem_bool;

namespace impl {

    template<typename T> struct is_letter_kind { static constexpr bool value = false; };
    template<char C> struct is_letter_kind<letter_kind<C>> { static constexpr bool value = true; };
    template<typename T> struct is_num_kind { static constexpr bool value = false; };
    template<int I> struct is_num_kind<num_kind<I>> { static constexpr bool value = true; };

    template<typename List, typename Acc>
    struct next_lexem;

    template<typename Acc>
    struct next_lexem<type_list<>, Acc>
    {
        using type = Acc;
        using list_tail = type_list<>;
    };

    template<typename... Cs, typename Acc>
    struct next_lexem<type_list<blank_kind, Cs...>, Acc>
    {
        static_assert(!std::is_same_v<Acc, nonesuch>);
        using type = Acc;
        using list_tail = type_list<Cs...>;
    };

    template<typename... Cs, typename Acc>
    struct next_lexem<type_list<close_paren, Cs...>, Acc>
    {
        static_assert(!std::is_same_v<Acc, nonesuch>);
        using type = Acc;
        using list_tail = type_list<close_paren, Cs...>;
    };

    template<typename... Cs>
    struct next_lexem<type_list<close_paren, Cs...>, nonesuch>
    {
        using type = close_paren;
        using list_tail = type_list<Cs...>;
    };

    template<typename... Cs>
    struct next_lexem<type_list<blank_kind, Cs...>, nonesuch>
    {
        using list_tail = typename next_lexem<type_list<Cs...>, nonesuch>::list_tail;
        using type = typename next_lexem<type_list<Cs...>, nonesuch>::type;
    };

    template<int I, int Acc, typename... Cs>
    struct next_lexem<type_list<num_kind<I>, Cs...>, lexem_int<Acc>>
    {
        using cur_acc = lexem_int<Acc * 10 + I>;
        using list_tail = typename next_lexem<type_list<Cs...>, cur_acc>::list_tail;
        using type = typename next_lexem<type_list<Cs...>, cur_acc>::type;
    };

    template<int I, typename... Cs>
    struct next_lexem<type_list<num_kind<I>, Cs...>, nonesuch>
    {
        using list_tail = typename next_lexem<type_list<Cs...>, lexem_int<I>>::list_tail;
        using type = typename next_lexem<type_list<Cs...>, lexem_int<I>>::type;
    };

    template<typename... Cs>
    struct next_lexem<type_list<blank_kind, Cs...>, lexem_ident<'#', 't'>>
    {
        using list_tail = type_list<Cs...>;
        using type = lexem_bool<true>;
    };

    template<typename... Cs>
    struct next_lexem<type_list<close_paren, Cs...>, lexem_ident<'#', 't'>>
    {
        using list_tail = type_list<close_paren, Cs...>;
        using type = lexem_bool<true>;
    };

    template<>
    struct next_lexem<type_list<>, lexem_ident<'#', 't'>>
    {
        using list_tail = type_list<>;
        using type = lexem_bool<true>;
    };

    template<typename... Cs>
    struct next_lexem<type_list<blank_kind, Cs...>, lexem_ident<'#', 'f'>>
    {
        using list_tail = type_list<Cs...>;
        using type = lexem_bool<false>;
    };

    template<>
    struct next_lexem<type_list<>, lexem_ident<'#', 'f'>>
    {
        using list_tail = type_list<>;
        using type = lexem_bool<false>;
    };

    template<typename... Cs>
    struct next_lexem<type_list<close_paren, Cs...>, lexem_ident<'#', 'f'>>
    {
        using list_tail = type_list<close_paren, Cs...>;
        using type = lexem_bool<false>;
    };

    template<char L, char... Acc, typename... Cs>
    struct next_lexem<type_list<letter_kind<L>, Cs...>, lexem_ident<Acc...>>
    {
        using list_tail = typename next_lexem<type_list<Cs...>, lexem_ident<Acc..., L>>::list_tail;
        using type = typename next_lexem<type_list<Cs...>, lexem_ident<Acc..., L>>::type;
    };

    template<typename... Cs>
    struct next_lexem<type_list<open_paren, Cs...>, nonesuch>
    {
        using type = open_paren;
        using list_tail = type_list<Cs...>;
    };

    template<char L, typename... Cs>
    struct next_lexem<type_list<letter_kind<L>, Cs...>, nonesuch>
    {
        using list_tail = typename next_lexem<type_list<Cs...>, lexem_ident<L>>::list_tail;
        using type = typename next_lexem<type_list<Cs...>, lexem_ident<L>>::type;
    };
}

template<typename L>
struct next_lexem;
template<typename... Cs>
struct next_lexem<type_list<Cs...>>
{
    using lexem = impl::next_lexem<type_list<Cs...>, nonesuch>;
    using type = typename lexem::type;
    using list_tail = typename lexem::list_tail;
};

template<>
struct next_lexem<type_list<>>
{
    using type = nonesuch;
    using list_tail = type_list<>;
};

static_assert(std::is_same_v<typename next_lexem<type_list<>>::type, nonesuch>);
static_assert(std::is_same_v<typename next_lexem<type_list<blank_kind, blank_kind, blank_kind>>::type, nonesuch>);
static_assert(std::is_same_v<typename next_lexem<type_list<num_kind<5>>>::type, lexem_int<5>>);
static_assert(std::is_same_v<typename next_lexem<type_list<num_kind<5>, num_kind<4>>>::type, lexem_int<54>>);
static_assert(std::is_same_v<typename next_lexem<type_list<blank_kind, blank_kind, num_kind<5>, num_kind<4>, close_paren>>::type, lexem_int<54>>);
static_assert(std::is_same_v<typename next_lexem<type_list<letter_kind<'#'>, letter_kind<'t'>>>::type, lexem_bool<true>>);
static_assert(std::is_same_v<typename next_lexem<type_list<letter_kind<'t'>, letter_kind<'r'>, letter_kind<'u'>>>::type, lexem_ident<'t', 'r', 'u'>>);
static_assert(std::is_same_v<typename next_lexem<type_list<letter_kind<'t'>, letter_kind<'r'>, letter_kind<'u'>, letter_kind<'e'>, letter_kind<'u'>>>::type, lexem_ident<'t', 'r', 'u', 'e', 'u'>>);

namespace impl
{
    template<typename LL, typename TL>
    struct to_lexem_list;
    template<typename LL>
    struct to_lexem_list<LL, type_list<>>
    {
        using type = LL;
    };
    template<typename... Ls, typename T, typename... Ts>
    struct to_lexem_list<type_list<Ls...>, type_list<T, Ts...>>
    {
        using lexem = ::next_lexem<type_list<T, Ts...>>;
        using type = typename mpl::If<std::is_same_v<typename lexem::type, nonesuch>,
            type_identity<type_list<Ls...>>,
            to_lexem_list<type_list<Ls..., typename lexem::type>, typename lexem::list_tail>>::type::type;
    };
}

template<typename List>
struct to_lexem_list;
template<typename... Cs>
struct to_lexem_list<type_list<Cs...>>
{
    using type = typename impl::to_lexem_list<type_list<>, type_list<Cs...>>::type;
};

static constexpr char test_lisp1[] = "  (  +   10   #f   ) ";
static_assert(std::is_same_v<
    typename to_lexem_list<
        typename mpl::list_map_from_literal<
            typename string_to_charlist<decltype(test_lisp1), test_lisp1>::type,
            box>::type
        >::type,
    type_list<
        open_paren,
        lexem_ident<'+'>,
        lexem_int<10>,
        lexem_bool<false>,
        close_paren
    >
>);

namespace impl
{
    template<typename L, typename LL>
    struct parse_list;
    
    template<typename... Ls, typename... LLs>
    struct parse_list<type_list<Ls...>, type_list<open_paren, LLs...>>
    {
        using sub_list = parse_list<type_list<>, type_list<LLs...>>;
        using next_atom = parse_list<type_list<Ls..., typename sub_list::type>, typename sub_list::list_tail>;
        using list_tail = typename next_atom::list_tail;
        using type = typename next_atom::type;
    };
    template<typename... Ls, typename LL, typename... LLs>
    struct parse_list<type_list<Ls...>, type_list<LL, LLs...>>
    {
        struct self
        {
            using type = type_list<Ls...>;
            using list_tail = type_list<LLs...>;
        };
        using next = typename mpl::If<std::is_same_v<LL, close_paren>,
                            self,
                            parse_list<type_list<Ls..., LL>, type_list<LLs...>>>::type;
        using list_tail = typename next::list_tail;
        using type = typename next::type;
    };

    template<typename LL>
    struct parse;
    template<>
    struct parse<type_list<>>
    {
        using type = nonesuch;
    };
    template<int I>
    struct parse<type_list<lexem_int<I>>>
    {
        using type = lexem_int<I>;
    };
    template<bool B>
    struct parse<type_list<lexem_bool<B>>>
    {
        using type = lexem_bool<B>;
    };
    template<char... Cs>
    struct parse<type_list<lexem_ident<Cs...>>>
    {
        using type = lexem_ident<Cs...>;
    };
    template<typename... Ls>
    struct parse<type_list<open_paren, Ls...>>
    {
        using type = typename impl::parse_list<type_list<>, type_list<Ls...>>::type;
    };
}

template<typename T, std::decay_t<T> V>
struct parse
{
    using type = typename impl::parse<
                    typename to_lexem_list<
                        typename mpl::list_map_from_literal<
                            typename string_to_charlist<T, V>::type,
                            box
                        >::type
                    >::type
                >::type;
};

static constexpr char test_lisp2[] = "  (  +   10  (953 let) #f ( * 54  21 )  ) ";
static constexpr char test_lisp3[] = "10";
static constexpr char test_lisp4[] = "let";
static constexpr char test_lisp5[] = "#t";
static_assert(std::is_same_v<
    typename parse<decltype(test_lisp2), test_lisp2>::type,
    type_list<
        lexem_ident<'+'>,
        lexem_int<10>,
        type_list<
            lexem_int<953>,
            lexem_ident<'l', 'e', 't'>
        >,
        lexem_bool<false>,
        type_list<
            lexem_ident<'*'>,
            lexem_int<54>,
            lexem_int<21>
        >
    >
>);
static_assert(std::is_same_v<
    typename parse<decltype(test_lisp3), test_lisp3>::type,
    lexem_int<10>
>);
static_assert(std::is_same_v<
    typename parse<decltype(test_lisp4), test_lisp4>::type,
    lexem_ident<'l', 'e', 't'>
>);
static_assert(std::is_same_v<
    typename parse<decltype(test_lisp5), test_lisp5>::type,
    lexem_bool<true>
>);

template<typename Operator, typename... Operands>
struct SExp;
enum class OpCode
{
    Add, Sub, Mul,
    Eq, Neq, Leq,
    Neg, Or, And,
    Not, Cons, Car,
    Cdr, IsNull,
    Let, Lambda,
    If
};
template <OpCode op> struct Op;
template<char...> struct Ident;
template<int I> struct Int;
template<bool B> struct Bool;
using False = Bool<false>;
using True = Bool<true>;
struct Nil;

template<typename Expr, typename Env>
struct eval;

template<typename Name, typename Value>
struct binding;

template<typename Ev, typename Bind, typename Env>
struct add_to_env;
template<typename... Es, char... Name, typename Value, typename Env, template<char...> class Identifier>
struct add_to_env<type_list<Es...>, type_list<Identifier<Name...>, Value>, Env>
{
    using type = type_list<binding<Ident<Name...>, typename eval<Value, Env>::type> , Es...>;
};
template<typename... Es, char... Name, typename Value, typename Env, template<char...> class Identifier>
struct add_to_env<type_list<Es...>, binding<Identifier<Name...>, Value>, Env>
{
    using type = type_list<binding<Ident<Name...>, typename eval<Value, Env>::type> , Es...>;
};

template <typename LetEnv, typename LetExpr> struct Let;
template <typename Params, typename LambdaExpr> struct Lambda;
template <typename Captures, typename Expr> struct Closure;
template <typename... Operands> struct Add;
template <typename... Operands> struct Sub;
template <typename... Operands> struct Mul;
template <typename... Operands> struct Eq;
template <typename... Operands> struct And;
template <typename... Operands> struct Or;
template <typename Operand> struct Not;


template<OpCode O, typename Env, typename... Ls>
struct to_evaluator;

template<typename Env, typename... Binds, typename Expr>
struct to_evaluator<OpCode::Let, Env, type_list<Binds...>, Expr>
{
    template<typename E, typename B>
    using real_add_to_env = add_to_env<E, B, Env>;

    using let_env = typename mpl::list_fold<type_list<Binds...>, Env, real_add_to_env>::type;

    using type = Let<let_env, Expr>;
};
template<typename Env, typename... Params, typename Expr>
struct to_evaluator<OpCode::Lambda, Env, type_list<Params...>, Expr>
{
    template <typename T> struct is_ident : std::false_type {};
    template <char... Cs> struct is_ident<lexem_ident<Cs...>> : std::true_type {};
    template <typename T> static constexpr bool is_ident_v = is_ident<T>::value;
    template <typename T> struct to_ident;
    template <char... Cs> struct to_ident<lexem_ident<Cs...>> { using type = Ident<Cs...>; };

    static_assert((is_ident_v<Params> && ...));
    using type = Lambda<type_list<typename to_ident<Params>::type...>, Expr>;
};

template<typename Env, typename... Ls>
struct to_evaluator<OpCode::Add, Env, Ls...>
{
    using type = Add<typename eval<Ls, Env>::type...>;
};
template<typename Env, typename... Ls>
struct to_evaluator<OpCode::Sub, Env, Ls...>
{
    using type = Sub<typename eval<Ls, Env>::type...>;
};
template<typename Env, typename... Ls>
struct to_evaluator<OpCode::Mul, Env, Ls...>
{
    using type = Mul<typename eval<Ls, Env>::type...>;
};
template<typename Env, typename... Ls>
struct to_evaluator<OpCode::Eq, Env, Ls...>
{
    using type = Eq<typename eval<Ls, Env>::type...>;
};
template<typename Env, typename L>
struct to_evaluator<OpCode::Not, Env, L>
{
    using type = Not<typename eval<L, Env>::type>;
};

template<bool B, typename Env>
struct eval<lexem_bool<B>, Env>
{
    using type = Bool<B>;
};
template<int I, typename Env>
struct eval<lexem_int<I>, Env>
{
    using type = Int<I>;
};
template<char... Name, typename Env>
struct eval<lexem_ident<Name...>, Env>
{
    template<typename T>
    struct finder;
    template<typename BindName, typename BindValue>
    struct finder<binding<BindName, BindValue>>
    {
        static constexpr bool cond_v = std::is_same_v<BindName, Ident<Name...>>;
        using type = BindValue;
    };
    
    using type = typename finder<typename mpl::list_find<Env, finder>::type>::type;
};
template<typename Env>
struct eval<type_list<>, Env>
{
    using type = Nil;
};
template<char... Name, typename... Ls, typename Env>
struct eval<type_list<lexem_ident<Name...>, Ls...>, Env>
{
    using op_eval = typename eval<lexem_ident<Name...>, Env>::type;
    using type = typename eval<type_list<op_eval, Ls...>, Env>::type;
};
template<OpCode O, typename... Ls, typename Env>
struct eval<type_list<Op<O>, Ls...>, Env>
{
    using evaluator_t = typename to_evaluator<O, Env, Ls...>::type;
    using type = typename eval<evaluator_t, Env>::type;
};
template<typename... Params, typename Expr, typename... Ls, typename Env>
struct eval<type_list<Lambda<type_list<Params...>, Expr>, Ls...>, Env>
{
    using type = typename eval<Closure<type_list<binding<Params, typename eval<Ls, Env>::type>...>, Expr>, Env>::type;
};
template<typename... Binds, typename Expr, typename Env>
struct eval<Closure<type_list<Binds...>, Expr>, Env>
{
    template<typename E, typename B>
    using real_add_to_env = add_to_env<E, B, Env>;

    using let_env = typename mpl::list_fold<type_list<Binds...>, Env, real_add_to_env>::type;
    using type = typename eval<Expr, let_env>::type;
};
template<typename L, typename... Ls, typename Env>
struct eval<type_list<L, Ls...>, Env>
{
    using type = type_list<typename eval<L, Env>::type, typename eval<Ls, Env>::type...>;
};
template<typename LetEnv, typename LetExpr, typename Env>
struct eval<Let<LetEnv, LetExpr>, Env>
{
    using type = typename eval<LetExpr, LetEnv>::type;
};
template<typename Params, typename Expr, typename Env>
struct eval<Lambda<Params, Expr>, Env>
{
    using type = Lambda<Params, Expr>;
};
template<int... Operands, typename Env>
struct eval<Add<Int<Operands>...>, Env>
{
    using type = Int<(Operands + ...)>;
};
template<int... Operands, typename Env>
struct eval<Mul<Int<Operands>...>, Env>
{
    using type = Int<(Operands * ...)>;
};
template<int Operand, typename Env>
struct eval<Sub<Int<Operand>>, Env>
{
    using type = Int<-Operand>;
};
template<bool Operand, typename Env>
struct eval<Not<Bool<Operand>>, Env>
{
    using type = Bool<!Operand>;
};
template<bool... Operands, typename Env>
struct eval<Eq<Bool<Operands>...>, Env>
{
    using type = Bool<(Operands && ...)>;
};
template<int I, typename Env>
struct eval<Int<I>, Env>
{
    using type = Int<I>;
};
template<bool B, typename Env>
struct eval<Bool<B>, Env>
{
    using type = Bool<B>;
};

using default_env = type_list<
    binding<Ident<'+'>, Op<OpCode::Add>>,
    binding<Ident<'*'>, Op<OpCode::Mul>>,
    binding<Ident<'-'>, Op<OpCode::Sub>>,
    binding<Ident<'e', 'q', 'u', 'a', 'l', '?'>, Op<OpCode::Eq>>,
    binding<Ident<'n', 'o', 't'>, Op<OpCode::Not>>,
    binding<Ident<'o', 'r'>, Op<OpCode::Or>>,
    binding<Ident<'a', 'n', 'd'>, Op<OpCode::And>>,
    binding<Ident<'i', 'f'>, Op<OpCode::If>>,
    binding<Ident<'l', 'e', 't'>, Op<OpCode::Let>>,
    binding<Ident<'l', 'a', 'm', 'b', 'd', 'a'>, Op<OpCode::Lambda>>
    >;

static constexpr char test_eval_lisp1[] = "(+ 10 953 54 21)";
static constexpr char test_eval_lisp2[] = R"--(
    (let ((x 2) (y 3))
        (* y x  (let ((z 21))
                    (+ y z)
                )
        )
    )
)--";
static constexpr char test_eval_lisp3[] = R"--(
    (let ((square (lambda (x)
                          (* x x))))
         (square 4))
)--";

int main()
{
    typename eval<typename parse<decltype(test_eval_lisp3), test_eval_lisp3>::type, default_env>::type::compile_error a;

    return 0;
}