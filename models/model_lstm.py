import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(n_out)

        self.relu2 = nn.ReLU(inplace = True) 
        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu2(out)
        return out
    
class DeepBrain(nn.Module):
    def __init__(self, block=BasicBlock, inplanes=27, planes=3, drop_out=True):
        super(DeepBrain, self).__init__()
        self.n_classes = 7
        
        
        
        
        inplanes=27
        planes=3
        self.preBlock = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, padding=0),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(planes, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True))
        
        self.layer_1 = self._make_layer(block,  24, 32, 2)
        self.layer_2 = self._make_layer(block, 32, 64, 2, pooling=True)
        self.layer_3 = self._make_layer(block, 64, 64, 2, pooling=True)
        self.layer_4 = self._make_layer(block, 64, 128, 2, pooling=True)
        
        self.post_conv = nn.Conv3d(128, 64, kernel_size=(5, 6, 6))
        
        if drop_out:
            self.classifier = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(64, self.n_classes),
                nn.LogSoftmax(dim=1))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.n_classes),
                nn.LogSoftmax(dim=1))            
        
        self._initialize_weights()
        
    def _make_layer(self, block, planes_in, planes_out, num_blocks, pooling=False, drop_out=False):
        layers = []
        if pooling:
#             layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            layers.append(block(planes_in, planes_out, stride=2))
        else:
            layers.append(block(planes_in, planes_out))
        for i in range(num_blocks - 1):
            layers.append(block(planes_out, planes_out))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self, x):
        
        x = self.preBlock(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.post_conv(x)
        x = x.view(-1, 64 * 1)
        
        x = self.classifier(x)
        
        return x


class DB_LSTM(nn.Module):
    def __init__(self,DB):
        super(DB_LSTM, self).__init__()
        self.model_DB=DB
        self.RNN = nn.LSTM(27,1)
        
    def forward(self,x):
        x=torch.transpose(x,1,-1)
        x=torch.reshape(x,(2,75* 93* 81,27))
        
        x=self.RNN(x)[0]
        x=torch.reshape(x,(2, 81, 75, 93, 1))
        x=torch.cat(tuple([x for i in range(27)]),-1)
        x=torch.transpose(x,1,-1)
        
        return self.model_DB(x)
    
    

    
    















    
class Feature_extractor(nn.Module):
    
    def __init__(self,inplanes,planes, block=BasicBlock):
        super(Feature_extractor, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, padding=0),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(planes, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True))
        
        self.layer_1 = self._make_layer(block,  24, 32, 2)
        self.layer_2 = self._make_layer(block, 32, 64, 2, pooling=True)
        self.layer_3 = self._make_layer(block, 64, 64, 2, pooling=True)
        self.layer_4 = self._make_layer(block, 64, 128, 2, pooling=True)
        
        self.post_conv = nn.Conv3d(128, 64, kernel_size=(5, 6, 6))
    
    def _make_layer(self, block, planes_in, planes_out, num_blocks, pooling=False, drop_out=False):
        layers = []
        if pooling:
#             layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            layers.append(block(planes_in, planes_out, stride=2))
        else:
            layers.append(block(planes_in, planes_out))
        for i in range(num_blocks - 1):
            layers.append(block(planes_out, planes_out))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self,x):
        x = self.preBlock(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.post_conv(x)
        x = x.view(-1, 64 * 1)
        return x











class  RNN(nn.Module):
    
    def __init__(self):
        super(RNN, self).__init__() 
        n_features=75* 93* 81
        len_size=27
        self.RNN = nn.LSTM(n_features, n_features,num_layers=len_size)
    
    def forward(self,x):
        

        x,_=self.RNN(x)
        
        return x
    

    
class  CF(nn.Module):
    def __init__(self, drop_out=False):#TODO
        super(CF, self).__init__()
        
        self.n_classes=7#TODO
        
        if drop_out:
            self.classifier = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(64, self.n_classes),
                nn.LogSoftmax(dim=1))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.n_classes),
                nn.LogSoftmax(dim=1))    
    
    def forward(self, x):
        
        return self.classifier(x)
    
    
class RNN_FEATURE_CF(nn.Module):
    def __init__(self):
        super(RNN_FEATURE_CF, self).__init__()
        self.feature_extractor=Feature_extractor(inplanes=1
                                               , planes=3)
        self.RNN=RNN()
        self.CF=CF()
         
        #self._initialize_weights
        #self.apply(self._initialize_weights)#apply to every submodule
        
    def forward(self,x):
        x=torch.transpose(x,0,1)
        x=torch.reshape(x,(2,27,75* 93* 81))
        x = self.RNN(x)
        
        x=torch.reshape(x,(2, 27, 81, 75, 93))
        x=torch.transpose(x,0,1)
        #x=torch.transpose(x,1,-1)
        x = self.feature_extractor(x)
        
        
        
        
        x = self.CF(x)
        
        return x


X=torch.rand(2, 27, 75, 93, 81)   
#model=RNN_FEATURE_CF()
model=nn.LSTM(100,100,1)


total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')
#model(X)