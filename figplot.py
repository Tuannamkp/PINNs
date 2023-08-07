import matplotlib.pyplot as plt
import torch
def plot_result(x,y,x_train,y_train,y_pre,i,x_phy=None):
    plt.figure(figsize=(8,4))
    plt.plot(x,y,color ="black",linewidth =2, alpha = 0.8, label = "True Value")
    plt.plot(x,y_pre,color ="tab:blue",linewidth = 4, alpha =0.8, label ="Normal NN Prediction")
    plt.scatter(x_train,y_train, s =60, color ="tab:orange",alpha =0.4, label="Training Data")
    if x_phy is not None:
        plt.scatter(x_phy, -0*torch.ones_like(x_phy),s=60,color ="tab:green",alpha =0.4, 
                    label = "PINN loss training location")
    l = plt.legend(loc=(1.01,0.34),frameon =False,fontsize ="large")
    plt.setp(l.get_texts(),color="k")
    plt.text(1.065,0.7,"Training step:%i"%(i+1),fontsize="xx-large",color="k")
    plt.axis("off")
