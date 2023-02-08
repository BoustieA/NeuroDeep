import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Basic Block of Deepbrain model
    Conv3D->BatchNorm->Relu->Conv3D->BatchNorm->Relu
    """
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
    
class Conv_Feature_extractor(nn.Module):
    """
    Chain of blocks of the DeepBrain model
    return the input of the classifier with forward
    """
    def __init__(self, block=BasicBlock, inplanes=27, planes=3, drop_out=True):
        super(Conv_Feature_extractor, self).__init__()
        
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
                    
        
        #self._initialize_weights()
        
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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self, x):
        
        x = self.preBlock(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.post_conv(x)
        
        return x


class DeepBrain(nn.Module):
    """
    Main DeepBrain model
    """
    def __init__(self, block=BasicBlock, inplanes=27, planes=3, drop_out=True):
        super(DeepBrain, self).__init__()
        self.n_classes = 7
        
        self.Feature_extractor = Conv_Feature_extractor(inplanes = inplanes, planes = planes
                                                        , drop_out = drop_out)
        
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
        
        self._initialize_weights()#no need to apply, access every submodule by default
        
    
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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self, x):
        
        x = self.Feature_extractor(x)
        x = x.view(-1, 64 * 1)
        
        x = self.classifier(x)
        
        return x
 
    
    def FT(self,FT_params):
        """
        
        Wrapper, prepare the model to fine_tune
        Update current object accordingly
        
        
        STEP 1 LOAD the MODEL
        STEP 2 adapt output or classifier for transfert learning
        STEP 3 freeze desired layers


        Parameters for fine tunning the model
        ----------
        
        parameters : Dictionnary 
            {"path_weights": path of pretrained weights,
             "reset": "all", "last_layer",
             "n_output":int /output dim of the last layer
             "freeze_type":"all","feature_extractor","progressive",
             "drop_out":Bool / usefull only if more than one layer is changed
             }
            
        
        Returns
        model : torch.nn model 
            pretrained model for transfert learning
            
            ready to train
            
            
        
        """
        self.load_model(FT_params["path_weights"])
            
        self.FT_adapt_classifier(FT_params["reset"],FT_params["n_output"],FT_params["drop_out"])
        self.FT_freeze_params(FT_params["freeze_type"])
        return self
    
    def load_model(self,path):
        """
        The model is modified in structure, since we slice a model
        into submodules. we iterate over the layers to get the right initialisation

        """
        pre_trained_model = torch.load(path,
                                         map_location=torch.device('cpu'))['state_dict']
        
        iter_=iter(pre_trained_model)
        for key in self.state_dict():
            if "Feature_extractor" in key:
                layer_name=key[18:]
                value=pre_trained_model.pop(layer_name)
                pre_trained_model[key]=value
        self.load_state_dict(pre_trained_model)
        
            
    def FT_adapt_classifier(self,reset, n_output, drop_out=False):
        """
        Modify the classifier for fine tune the model
        #replace both or just the last layer then initialize_weights
        
        reset:
        all: all layers are changed
        last_layer : last layer is changed
        None: No change
              
        """
        
        if reset=="all":
            if drop_out:
                self.classifier = nn.Sequential(
                    nn.Linear(64, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(64, n_output),
                    nn.LogSoftmax(dim=1))
                
                
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(64, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, n_output),
                    nn.LogSoftmax(dim=1))
            # weights initialisation    
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
                    
        elif reset=="last_layer":#TODO
            self.classifier[-2]=nn.Linear(self.classifier[-2].in_features,
                                                    n_output)
            # weights initialisation
            self.classifier[-2].weight.data.normal_(0, 0.01)
            self.classifier[-2].bias.data.zero_()
        else:
            pass
        
        
        
    def FT_freeze_params(self,freeze_type):
        """
        freeze layers to prevent updating weights
        all: all layers but the last one
        feature_extractor : only the layers of the Feature extractor are frozen
        progressive:learning rate decaying with deepness#TODO
        None: No freeze

        """
        
        if freeze_type=="all":
            for param in self.parameters():#freeze all layers
                param.requires_grad = False
            self.classifier[-2].weight.requires_grad = True #unfreeze last layer
            self.classifier[-2].bias.requires_grad = True #unfreeze last layer
            
        elif freeze_type=="feature_extractor":
            #freeze only
            for param in self.Feature_extractor.parameters():
                param.requires_grad = False
        elif freeze_type=="progressive":
            pass
            
        