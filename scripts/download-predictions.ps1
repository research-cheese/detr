# scp -r ataparia@en4228255l.cidse.dhcp.asu.edu:Documents/NOEL/detr/predictions/train ./scp/predictions/train

mkdir -p scp/aerial/dust-10/train/annotations
scp -r ataparia@en4228255l.cidse.dhcp.asu.edu:Documents/NOEL/detr/aerial/dust-10/train/annotations ./scp/aerial/dust-10/train/annotations
scp -r ataparia@en4228255l.cidse.dhcp.asu.edu:Documents/NOEL/detr/aerial/dust-10/train/metadata ./scp/aerial/dust-10/train/metadata
scp -r ataparia@en4228255l.cidse.dhcp.asu.edu:Documents/NOEL/detr/aerial/dust-10/train/train2017 ./scp/aerial/dust-10/train/train2017

