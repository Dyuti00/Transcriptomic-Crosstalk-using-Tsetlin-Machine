# ================================
# BIOINFORMATICS PIPELINE
# T2D vs PANCREATIC CANCER
# ================================

library(DESeq2)
library(dplyr)

# -------- FILE PATHS (CHANGE LATER IF NEEDED) --------
cancer_counts <- "GSE79668_counts_with_GSM.csv"
cancer_meta   <- "metadata_GSE79668.csv"

t2d_counts <- "GSE164416_EXP_FILE.csv"
t2d_meta   <- "meta_data_new.csv"

# -------- LOAD DATA --------
cancer_data <- read.csv(cancer_counts, row.names = 1)
cancer_meta <- read.csv(cancer_meta, row.names = 1)

t2d_data <- read.csv(t2d_counts, row.names = 1)
t2d_meta <- read.csv(t2d_meta, row.names = 1)

# -------- CLEAN --------
cancer_data <- round(cancer_data)
t2d_data <- round(t2d_data)

# -------- DGE: CANCER --------
dds_cancer <- DESeqDataSetFromMatrix(
  countData = cancer_data,
  colData = cancer_meta,
  design = ~ condition
)

dds_cancer <- DESeq(dds_cancer)
res_cancer <- results(dds_cancer)

res_cancer <- res_cancer[which(res_cancer$padj < 0.05 & abs(res_cancer$log2FoldChange) > 1),]
write.csv(as.data.frame(res_cancer), "Cancer_DEGs.csv")

# -------- DGE: T2D --------
dds_t2d <- DESeqDataSetFromMatrix(
  countData = t2d_data,
  colData = t2d_meta,
  design = ~ condition
)

dds_t2d <- DESeq(dds_t2d)
res_t2d <- results(dds_t2d)

res_t2d <- res_t2d[which(res_t2d$padj < 0.05 & abs(res_t2d$log2FoldChange) > 1),]
write.csv(as.data.frame(res_t2d), "T2D_DEGs.csv")

# -------- OVERLAP --------
cancer <- read.csv("Cancer_DEGs.csv")
t2d <- read.csv("T2D_DEGs.csv")

cancer$gene <- rownames(cancer)
t2d$gene <- rownames(t2d)

overlap <- inner_join(cancer, t2d, by="gene")

write.csv(overlap, "Common_DEGs.csv")

cat("Number of common DEGs:", nrow(overlap))

library(clusterProfiler)
library(org.Hs.eg.db)

genes <- read.csv("Common_DEGs.csv")
gene_list <- genes$gene

# Convert to ENTREZ
entrez <- bitr(gene_list, fromType="SYMBOL",
               toType="ENTREZID",
               OrgDb="org.Hs.eg.db")

# KEGG
kegg <- enrichKEGG(gene = entrez$ENTREZID, organism = 'hsa')

write.csv(as.data.frame(kegg), "KEGG_results.csv")
