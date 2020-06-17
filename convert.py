from pyensae.languages import r2python

rscript = """
temp <- data[clust==i,]
"""
print(r2python(rscript, pep8=True))
