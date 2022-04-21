class SceneData:
    
    def __init__(self, nw, nf, ew, ef, eu, ev):
        self.node_words = nw
        self.node_features = nf
        self.edge_words = ew
        self.edge_features = ef
        self.edges_u = eu
        self.edges_v = ev