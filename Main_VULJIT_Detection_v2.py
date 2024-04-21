import argparse
from jitvul_model.jit_vul_detection_model import  *
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.utils import remove_isolated_nodes
from torch_geometric.utils import to_networkx
from torch_geometric.utils import degree
from collections import Counter
from jitvul_model.GATv2Conv import *

def train_model(model, _trainLoader, starting_epochs = 0):
    torch.manual_seed(12345)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = model.optimizer
    criterion = torch.nn.CrossEntropyLoss()
    starting_epochs += 1
    valid_auc = 0
    last_train_loss = -1
    last_acc = 0
    for e in range(starting_epochs, 5):
        train_loss, acc = train_v2(e, _trainLoader, model, criterion, optimizer, device)

def train_v2(curr_epochs, _trainLoader, model, criterion, optimizer, device):
    train_loss = 0
    correct = 0
    model.train()
    for graph, commit_id, index in _trainLoader:
        if graph.num_nodes > 1500:
            graph =  graph.subgraph(torch.LongTensor(list(range(0, 1500))))
        # if index % 500 == 0:
        #     print("curr: {}".format(index) + " train loss: {}".format(train_loss / (index + 1)) + " acc:{}".format(correct / (index + 1)))
        if device != torch.device('cpu'):
            
            graph = graph.cuda()
        # if graph.y.item() == 1:
        #     target = torch.tensor([[0,1]],dtype=float)
        # else:
        #     target = torch.tensor([[1,0]],dtype=float)
        # #if graph.y 
        target = graph.y
        target = target.to(device)
        if graph.num_nodes == 0 or graph.num_edges == 0:
            continue
        out = model(graph.x, graph.edge_index)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = out.max(1)
        # print('pred',predicted)
        # print('-'*3)
        correct += predicted.eq(target).sum().item()
        del graph.x, graph.edge_index, graph.edge_type, graph.y, graph, predicted, out
    avg_train_loss = train_loss / len(_trainLoader)
    acc = correct / len(_trainLoader)
    print("correct:", correct)
    print("epochs {}".format(curr_epochs) + " train loss: {}".format(avg_train_loss) + " acc: {}".format(acc))
    return avg_train_loss, acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', type=str, help='dir of graph data')
    parser.add_argument('--train_file', type=str, help='file of training data')
    parser.add_argument('--test_file', type=str, help='file of testing data')
    parser.add_argument('--model_dir', type=str, help='output of trained model',
                        default='Model')
    parser.add_argument('--model_name', type=str, help='name of the model',
                        default='best_model')
    parser.add_argument('--GNN_type', default= "RGCN")
    parser.add_argument('--graph_readout_func', default= "add")
    parser.add_argument('--mode', default=  "train_only")
    parser.add_argument('--hidden_size', default=  32)
    parser.add_argument('--learning_rate', default=  0.00001)
    parser.add_argument('--dropout_rate', default= 0.2)
    parser.add_argument('--max_epochs', default= 10)
    parser.add_argument('--num_of_layers', default= 2)

    args = parser.parse_args()

    graph_path = args.graph_dir
    train_path = args.train_file
    test_path = args.test_file
    model_path = args.model_dir
    mode = args.mode
    params = {'max_epochs': int(args.max_epochs), 'hidden_size': int(args.hidden_size), 'lr': float(args.learning_rate), 'dropout_rate': float(args.dropout_rate),
              "num_of_layers": int(args.num_of_layers), 'model_name': args.model_name, 'GNN_type': args.GNN_type, "graph_readout_func": args.graph_readout_func}
    torch.manual_seed(12345)
    tmp_file = open(train_path, "r").readlines()
    train_files = [f.replace("\n", "") for f in tmp_file]
    print(len(train_files))

    train_dataset = GraphDataset(train_files, graph_path)
    _trainLoader = DataLoader(train_dataset, collate_fn=collate_batch, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_epochs = int(args.max_epochs)

    data = {}
    for  graph, _, index in _trainLoader:
        data = graph
        break
    print(data)


    # Print information about the dataset
    print(f'Dataset: {data}')
    print('-------------------')
    print(f'Number of graphs: {len(data)}')
    print(f'Number of nodes: {data.x.shape[0]}')
    print(f'Number of features: {data.num_features}')
    print(f'Number of classes: {data.is_directed}')
    isolated = (remove_isolated_nodes(data['edge_index'])[2] == False).sum(dim=0).item()
    print(f'Number of isolated nodes = {isolated}')

    # G = to_networkx(data, to_undirected=True)
    # plt.figure(figsize=(18,18))
    # plt.axis('off')
    # print(G)
    # nx.draw_networkx(G,
    #                 pos=nx.spring_layout(G, seed=0),
    #                 with_labels=False,
    #                 node_size=50,
    #                 node_color='grey',
    #                 width=2,
    #                 edge_color="red"
    #                 )
    # plt.show()

    # Get list of degrees for each node
    degrees = degree(data.edge_index[0]).numpy()

    # Count the number of nodes for each degree
    numbers = Counter(degrees)

    # # Bar plot
    # fig, ax = plt.subplots(figsize=(18, 7))
    # ax.set_xlabel('Node degree')
    # ax.set_ylabel('Number of nodes')
    # plt.bar(numbers.keys(),
    #         numbers.values(),
    #         color='#0A047A')
    # plt.show()

    gcn = GCN(data.num_node_features, 32, 2)
    print(gcn)
    gatc = GATv2(data.num_node_features, 32, 2, dropout=0.2)
    print(gatc)
    train_model(gatc, _trainLoader)
# python3 Main_VULJIT_Detection_v2.py --graph_dir='Data/Embedding'  --train_file='Data/data_split/cross_train.txt' --test_file='Data/data_split/cross_test.txt'  --model_dir='Model' --GNN_type="GAT"  --model_name="gat_v2" --mode="train_and_test"     