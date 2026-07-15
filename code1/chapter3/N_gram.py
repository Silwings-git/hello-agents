# 示例语料库
import collections


corpus = "datawhale agent learns datawhale agent works"
tokens = corpus.split()
total_tokens = len(tokens)

# 第一步: 计算 P(datawhale)
count_datawhale = tokens.count("datawhale")
p_datawhale = count_datawhale / total_tokens
print(f"第一步: P(datawhale) = {count_datawhale}/{total_tokens}")

# 第二步: 计算 P(agent|datawhale)
# 先计算 bigrams
bigrams = zip(tokens, tokens[1:])
bigrams_counts = collections.Counter(bigrams)
count_datawhale_agent = bigrams_counts[("datawhale", "agent")]
p_agent_given_datawhale = count_datawhale_agent / count_datawhale
print(
    f"第二步: P(agent|datawhale) = {count_datawhale_agent}/{count_datawhale} = {p_agent_given_datawhale:.3f}"
)

# 第三步: 计算 P(learns|agent)
count_agent_learns = bigrams_counts[("agent", "learns")]
count_agent = tokens.count("agent")
p_learns_given_agent = count_agent_learns / count_agent
print(
    f"第三步: P(learns|agent) = {count_agent_learns}/{count_agent} = {p_learns_given_agent:.3f}"
)

# 将概率连乘
p_sentence = p_datawhale * p_agent_given_datawhale * p_learns_given_agent
print(
    f"最后: P('datawhale agent learns') ≈ {p_datawhale:.3f} * {p_agent_given_datawhale:.3f} * {p_learns_given_agent:.3f} = {p_sentence:.3f}"
)
