import os


def make_tsv(metadata, save_path):
    metadata = [str(x) for x in metadata]
    with open(os.path.join(save_path, 'metadata.tsv'), 'w') as f:
        for x in metadata:
            f.write(x + '\n')


# https://github.com/tensorflow/tensorboard/issues/44 image label will be squared
def make_sprite(label_img, save_path):
    import math
    import torch
    import torchvision
    from .x2num import makenp
    # this ensures the sprite image has correct dimension as described in
    # https://www.tensorflow.org/get_started/embedding_viz
    nrow = int(math.ceil((label_img.size(0)) ** 0.5))

    label_img = torch.from_numpy(makenp(label_img))  # for other framework
    # augment images so that #images equals nrow*nrow
    label_img = torch.cat((label_img, torch.randn(nrow ** 2 - label_img.size(0), *label_img.size()[1:]) * 255), 0)

    torchvision.utils.save_image(label_img, os.path.join(save_path, 'sprite.png'), nrow=nrow, padding=0)


def append_pbtxt(metadata, label_img, save_path, global_step, tag):
    with open(os.path.join(save_path, 'projector_config.pbtxt'), 'a') as f:
        # step = os.path.split(save_path)[-1]
        f.write('embeddings {\n')
        f.write('tensor_name: "{}:{}"\n'.format(tag, global_step))
        f.write('tensor_path: "{}"\n'.format(os.path.join(global_step, 'tensors.tsv')))
        if metadata is not None:
            f.write('metadata_path: "{}"\n'.format(os.path.join(global_step, 'metadata.tsv')))
        if label_img is not None:
            f.write('sprite {\n')
            f.write('image_path: "{}"\n'.format(os.path.join(global_step, 'sprite.png')))
            f.write('single_image_dim: {}\n'.format(label_img.size(3)))
            f.write('single_image_dim: {}\n'.format(label_img.size(2)))
            f.write('}\n')
        f.write('}\n')


def make_mat(matlist, save_path):
    with open(os.path.join(save_path, 'tensors.tsv'), 'w') as f:
        for x in matlist:
            x = [str(i) for i in x]
            f.write('\t'.join(x) + '\n')
