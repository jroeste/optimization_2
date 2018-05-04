# optimization_2, endringer

Endra til å alltid starte et gyldig sted, midt mellom lambda high og low
La til alternativ end condition i tillegg til KKT, at den slutter hvis alpha blir 1e-100
Endra fra min() til amin()
Endra fra max(df - columnvector) til np.norm(.,2)

# Disse reserve - exit conditions har vi:
BFGS gir opp å tilfredsstille grad p mindre enn tau om alpha blir mindre enn 1e-5
Gir opp å tilfredsstille 

# Spørsmål
Vil BFGS alltid gi oss en descent-retning?
