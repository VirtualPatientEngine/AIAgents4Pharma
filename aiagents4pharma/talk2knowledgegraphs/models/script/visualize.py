import os
import sys
import pprint

import torch
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util


vocab_file = os.path.join(os.path.dirname(__file__), "../data/fb15k237_entity.txt")
vocab_file = os.path.abspath(vocab_file)


def load_vocab(dataset):
    entity_mapping = {}
    with open(vocab_file, "r") as fin:
        for line in fin:
            k, v = line.strip().split("\t")
            entity_mapping[k] = v
    entity_vocab = []
    with open(os.path.join(dataset.raw_dir, "entities.dict"), "r") as fin:
        for line in fin:
            id, e_token = line.strip().split("\t")
            entity_vocab.append(entity_mapping[e_token])
    relation_vocab = []
    with open(os.path.join(dataset.raw_dir, "relations.dict"), "r") as fin:
        for line in fin:
            id, r_token = line.strip().split("\t")
            relation_vocab.append("%s (%s)" % (r_token[r_token.rfind("/") + 1:].replace("_", " "), id))

    return entity_vocab, relation_vocab


def visualize_path(model, test_data, triplet, entity_vocab, relation_vocab, filtered_data=None):
    num_relation = len(relation_vocab)
    triplet = triplet.unsqueeze(0)
    inverse = triplet[:, [1, 0, 2]]
    inverse[:, 2] += num_relation
    model.eval()
    t_batch, h_batch = tasks.all_negative(test_data, triplet)
    t_pred = model(test_data, t_batch)
    h_pred = model(test_data, h_batch)

    if filtered_data is None:
        t_mask, h_mask = tasks.strict_negative_mask(test_data, triplet)
    else:
        t_mask, h_mask = tasks.strict_negative_mask(filtered_data, triplet)
    pos_h_index, pos_t_index, pos_r_index = triplet.unbind(-1)
    t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask).squeeze(0)
    h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask).squeeze(0)

    logger.warning("")
    samples = (triplet, inverse)
    rankings = (t_ranking, h_ranking)
    for sample, ranking in zip(samples, rankings):
        h, t, r = sample.squeeze(0).tolist()
        h_name = entity_vocab[h]
        t_name = entity_vocab[t]
        r_name = relation_vocab[r % num_relation]
        if r >= num_relation:
            r_name += "^(-1)"
        logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.warning("rank(%s | %s, %s) = %g" % (t_name, h_name, r_name, ranking))

        paths, weights = model.visualize(test_data, sample)
        for path, weight in zip(paths, weights):
            triplets = []
            for h, t, r in path:
                h_name = entity_vocab[h]
                t_name = entity_vocab[t]
                r_name = relation_vocab[r % num_relation]
                if r >= num_relation:
                    r_name += "^(-1)"
                triplets.append("<%s, %s, %s>" % (h_name, r_name, t_name))
            logger.warning("weight: %g\n\t%s" % (weight, " ->\n\t".join(triplets)))


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))

    if cfg.dataset["class"] != "FB15k-237":
        raise ValueError("Visualization is only implemented for FB15k237")

    dataset = util.build_dataset(cfg)
    cfg.model.num_relation = dataset.num_relations
    model = util.build_model(cfg)
    entity_vocab, relation_vocab = load_vocab(dataset)

    device = util.get_device(cfg)
    model = model.to(device)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    # use the whole dataset for filtered ranking
    filtered_data = Data(edge_index=dataset.data.target_edge_index, edge_type=dataset.data.target_edge_type)
    filtered_data = filtered_data.to(device)

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    for i in range(500):
        visualize_path(model, test_data, test_triplets[i], entity_vocab, relation_vocab, filtered_data=filtered_data)
