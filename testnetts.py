import pandas as pd
import networkx as nx


def get_dipms(q=0.2):
    df = pd.read_csv("Output/adj_list_PFDN2_2.txt", sep="\t")
    df2 = pd.read_csv("Output/adj_List_uxt_3.txt", sep="\t")
    df = pd.concat([df, df2])
    #df = df[df["confidence"].isin(["High confidence"])]
    #df = df[abs(df['Frequency_crapome_ProtA'] -
   # df['Frequency_crapome_ProtB'])<q]
    df = df[~df['ProtA'].str.contains('DECOY')]
    df = df[~df['ProtB'].str.contains('DECOY')]
    return nx.from_pandas_edgelist(df, source="ProtA", target="ProtB", edge_attr=["CombProb"])

G = get_dipms()


ppi = pd.read_csv('PFDNppi.txt', sep='\t')
G2 = nx.from_pandas_edgelist(ppi, source='source', target='target')

tt = []
for x in G2.edges():
    if G.has_edge(*x):
        tt.append([x[0], x[1],G.edges[x]['CombProb']])

tt = pd.DataFrame(tt)
print(tt.shape)
tt.to_csv('ppi_scored.csv')
