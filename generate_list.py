import os

data_root = '/opt/dataset/saliency/MSRA-B/Imgs/'
with open('images.lst') as f:
    lines = f.readlines()

outfile = open('train_edge.lst', 'w')
for line in lines:
    line = line[:-1]
    im_name = data_root + line
    gt_name = data_root + line[:-4] + '.png'
    if not os.path.exists(im_name):
        print im_name
    if not os.path.exists(gt_name):
        print gt_name
    outfile.write(im_name + ' ' + gt_name + '\n')
outfile.close()
