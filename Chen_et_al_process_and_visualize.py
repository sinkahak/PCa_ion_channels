import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from scipy.io import mmread 
import random
import seaborn as sns
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

random.seed(321)

# fetches current wd
datadir = Path.cwd() 

counts = pd.read_csv(f'{datadir}/inputs/GSM4203181_data.raw.matrix.txt.gz', compression='gzip', delimiter='\t', index_col=0)

adata = sc.AnnData(counts.T)
adata.var_names = counts.index
adata.obs_names = counts.columns

sc.pp.filter_cells(adata, min_genes=600)
sc.pp.filter_genes(adata, min_cells=10)

# adata = adata[adata.obs.n_genes < 40000, :].copy()

# Reformat matrix to csr
adata.X = adata.X.tocsr()

# back up the raw UMI count in a layer called "counts"
adata.layers["counts"] = adata.X.copy()
adata.write(f'{datadir}/anndata/scRNA_Chen_adata.h5ad')

# select top n variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")

sc.pl.highly_variable_genes(adata, log=True)
plt.savefig(f'{datadir}/figures/hvg_plot_scRNA_chen.pdf')
plt.tight_layout()
plt.close()

# normalize and scale
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)

# PCA
sc.tl.pca(adata, n_comps=30, svd_solver="auto")
sc.pp.neighbors(adata, metric="cosine")

# visualize the PCA rank plot
sc.pl.pca_variance_ratio(adata, log=True, show=False)
plt.savefig(f'{datadir}/figures/pca_rankplot_scRNA_chen.pdf')
plt.tight_layout()
plt.close()

# Cluster, first do a check with a few clustering resolutions
resolutions = [0.1,0.2, 0.4, 0.6, 0.8, 1.0]
n_clusters = []
for res in resolutions:
    sc.tl.leiden(adata, resolution=res)
    n_clusters.append(adata.obs['leiden'].nunique())

plt.plot(resolutions, n_clusters, marker='o')
plt.xlabel('Resolution')
plt.ylabel('Number of clusters')
plt.tight_layout()
plt.savefig(f'{datadir}/figures/chen_clustering_resolution.pdf')
plt.close()

# Clustering with resolution = 0.2
sc.tl.leiden(adata, resolution = 0.2)

# visualize the UMAP with clusters
sc.tl.umap(adata)
sc.pl.umap(adata, color="leiden",show=False)
plt.tight_layout()
plt.savefig(f'{datadir}/figures/umap_scRNA_chen_clusters.pdf')
plt.close()

adata.write(f'{datadir}/anndata/scRNA_Chen_adata_clustered.h5ad')


#### Annotate the cell types ####
# subcluster based on the cluster 10
sc.tl.leiden(adata, resolution = 0.1, restrict_to = ('leiden', ['10']))

sc.pl.umap(adata, color='leiden_R', show=False)
plt.tight_layout()
plt.savefig(f'{datadir}/figures/umap_scRNA_chen_clusters_subclustered.pdf')
plt.close()


markers = pd.read_csv(f'{datadir}/cell_markers_for_plotting_mod.csv', sep=';')
annot = markers.groupby('cell_type_toPlot')['gene_id_hgnc'].apply(list).to_dict()

cell_types_to_ignore = ['Neuro', 'Neutro', 'Macro'] 

markers = markers[
    markers['gene_id_hgnc'].isin(set(adata.var)) &  # only keep genes in adata
    ~markers['cell_type_toPlot'].isin(cell_types_to_ignore)  # keep only specific cell types
]

# group into a dictionary for plotting
annot = markers.groupby('cell_type_toPlot')['gene_id_hgnc'].apply(list).to_dict()

dp = sc.pl.DotPlot(adata,var_names=annot, groupby='leiden_R',standard_scale="var")
dp.legend(colorbar_title='Mean expression')
dp.style(cmap='coolwarm')
dp.savefig(f'{datadir}/figures/scRNA_Chen_dotplot.pdf',bbox_inches='tight')

# add cell type info
celltypes = adata.obs['leiden_R'].copy()
celltypes = celltypes.replace(['0','2','3','4','5','8','10,0','13','14','16','18'],'tumor/luminal')
celltypes = celltypes.replace(['10,1'],'basal')
celltypes = celltypes.replace(['10,2'],'club')
celltypes = celltypes.replace(['6','11','17'],'endothelium')
celltypes = celltypes.replace(['1'],'T')
celltypes = celltypes.replace(['7'],'fibroblast/muscle')
celltypes = celltypes.replace(['15'],'B')
celltypes = celltypes.replace(['9'],'myeloid')
celltypes = celltypes.replace(['12'],'mast')
adata.obs['celltypes'] = celltypes

sc.pl.umap(adata, color=['celltypes'],size=5,palette='tab10_r',show=False)
plt.savefig(f'{datadir}/figures/umap_scRNA_chen_celltypes.pdf')

# broad cell types:
adata.obs['celltypes_broad'] = adata.obs['celltypes'].apply(
    lambda x: 'epithelium' if x in ['tumor/luminal', 'club', 'basal'] else 
               'immune' if x in ['myeloid', 'T', 'B', 'mast'] else
                'other_stroma' if x in ['fibroblast/muscle','endothelium'] else x)


#### visualization of the ion channels
ion_channels = pd.read_csv(f'{datadir}/ion_channels_subset_to_plot.csv', sep=';')
ion_channels_sub["group"] = ion_channels_sub["group"].fillna("Ungrouped").replace("", "Ungrouped").copy()

# ion_channels_sub = ion_channels_sub.groupby("group")["gene"].apply(list).to_dict()

genes_not_present = list(set(ion_channels['gene']) - set(adata.var_names))
gene_list_filt = list(set(ion_channels['gene']) - set(genes_not_present))

ion_channels_sub_grouped = (
    ion_channels_sub[ion_channels_sub["gene"].isin(gene_list_filt)]
    .groupby("group")["gene"]
    .apply(list) # put all values in a group to a list
    .to_dict()
)


# first ignore the genes that are not present to avoid the error with sc.pl_dotplot
genes_not_present = list(set(gene_list) - set(adata.var_names))
gene_list_filt = list(set(gene_list) - set(genes_not_present))

# plot without scaling first
dp1=sc.pl.dotplot(adata, var_names=gene_list_filt, groupby='celltypes', dendrogram=True, show=False, return_fig=True)
dp1.style(cmap='flare')
dp1.savefig(f'{datadir}/figures/Chen_scRNA_dotplot_ion_channels_subset_celltype.pdf', dpi=300, pad_inches=0.5)

# do scaling
dp2=sc.pl.dotplot(adata, var_names=gene_list_filt, groupby='celltypes', dendrogram=True, show=False, return_fig=True, standard_scale='var')
dp2.style(cmap='flare')
dp2.savefig(f'{datadir}/figures/Chen_scRNA_dotplot_ion_channels_subset_celltype_scaled.pdf', dpi=300, pad_inches=0.5)
