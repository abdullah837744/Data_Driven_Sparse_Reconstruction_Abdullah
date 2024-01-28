clc
close all
clear all

%%%%%%%%%%%%
F_Platform      = 2 ;

	if (F_Platform == 1 )
        cd ('D:\Chen LU\Abdullah\Gram_Schimdt\K_ELM_95\K_2_20')
        addpath ('D:\Chen LU\SparReconTestCase\ChenLuLibrary')
        addpath ('D:\Chen LU\Hard Disk\SimulationData\CylinderData\deltaT=0.2s\Re100t')
        addpath ('D:\Chen LU\Abdullah\ELM AutoEncoder\ELM_Library')

        
        
	elseif (F_Platform == 2 )
		cd ('/home/cfsr123/Abdullah/Gram_Schimdt/Results')
		addpath ('/home/cfsr123/Abdullah/ELM_Library')
		addpath ('/home/cfsr123/Abdullah/Input')
        addpath ('/home/cfsr123/Abdullah/ChenLuLibrary')
	end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    fstart        = 1501               ;       % starting file  number
    fend          = 1800             ;       % ending file  number
    finc          = 1               ;       % file increment
    iname         = 'Input_'         ;       % input file name
    iext          = '.in'           ;       % input file name extension
    lowertest     = 1               ;       % Repairing starting snapshot
	uppertest     = 300             ;       % Repairing ending snapshot
	testinc       = 1               ;       % Repairing snapshot increment



    H95           = 6              ;
    M = 0                           ;
    delimiterIn   = ''              ;       % Delimiter of the data
    headerlinesIn = 2               ;       % Headerline number of the data file
    columnheader  = 0               ;       % Columnline number of the data file
    scalar        = 2               ;       % 1 is u, 2 is u & v, 3 is u,v,w
    dimen         = 2               ;       % 1 is x, 2 is x & y, 3 is x,y,z

    Eseed = 101:120 ;
    
%% Read Snapshot data

    for i=fstart:finc:fend
		M=M+1;
    end
    
    for i=fstart:finc:fend
        disp(['Collecting data: ', int2str(i)]);
        rawdata=dlmread([iname int2str(i) iext],delimiterIn,headerlinesIn,columnheader);
        if i == fstart
            [nrow,~]      = size(rawdata)     ; 
            X = zeros(nrow*2, M)              ;  
        end   
        X(:,i-fstart+1)=reshape(rawdata(:,3:4),[],1); 
    end
    nrow2 = nrow*2;
   
    
    	%Take the mean out
	tempmean=mean(X,2);
	OriginX=X;
	for i=1:M
		X(:,i)=X(:,i)-tempmean;
    end
    
rand_seed = 101 ;
    
%Read_Num_H = dlmread('x_ELM_predict.out',',',1,0);
Num_H = 6:6:36;
points = 6:6:72;

    MSE_Exact_ELM_Gram = zeros(length(Num_H),2);
    MSE_Recon_ELM_Gram = zeros(length(Num_H),length(points));
    

for k = Num_H ;

modes = k   

A = [X' X'];

 [Train_Time,Train_Accuracy,Theta_1,Theta_2,Z_train,Predicted_X] = ELM_seed_theta1(A,0,k,'radbas',nrow2,rand_seed);


 
 
%% Gram_Schimdt_Orthogonalizaton

U = Theta_2' ;
V = zeros(size(U)) ;
V(:,1) = U(:,1) ;

for t = 2:k
    
    Proj = 0 ;
    for j = 2:t
        Proj = Proj + ((U(:,t)'*V(:,j-1)) / (V(:,j-1)'*V(:,j-1))).*V(:,j-1) ;
    end
    
    V(:,t) = U(:,t) - Proj ;
    
end
%%% Normalization
for m = 1:k
    
    V(:,m) = V(:,m)/(sqrt(V(:,m)'*V(:,m))) ;
end

Theta_2_nor = V' ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Later Analysis


Z_gram = (X'*(pinv(Theta_2_nor)))' ;
X_ELM_Recon = (Z_gram' * Theta_2_nor)' ;

for j = 1:M
    X_ELM_Recon(:,j) = X_ELM_Recon(:,j) + tempmean ;
end

ELM_Exact_Recon_Err = OriginX - X_ELM_Recon ;
MSE_Exact_Recon = sqrt(sum(sum(ELM_Exact_Recon_Err.^2)))/(nrow2*M) ;

MSE_Exact_ELM_Gram (k/6,1) = k ;
MSE_Exact_ELM_Gram (k/6,2) = MSE_Exact_Recon ;











%% Sparse Reconstruction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MSE_Sparse_p = zeros(1,length(points));
for p = points
    pnt = p/6;    
    disp(['Computing for point: ', int2str(pnt)]);    
  
%% Construct mask vector
    meas = p;        % Number of measurements
    unmeas=nrow-meas;
    
    rng('default');
    rng(101)
    randnum = randperm(nrow)             ;
    index   = randnum(1:meas)            ;
    index2  = randnum(meas+1:nrow)       ;
    
    mask       = zeros(nrow*2,M) ;
    NewX       = zeros(nrow*2,M) ;
    
    for j = 1:M
    mask(index,j)= 1               ;
    mask(index+nrow,j)=1           ;
    end
    
    Xtilde = mask.*X;

 

%% Reconstructing
C = mask';


Recon_X = zeros(nrow2,M);




MSE = 0 ;
for j = 1:M

   
     for i = 1:k
         C_Theta_2_nor (i,:)=C(j,:).*Theta_2_nor (i,:);
         
     end
    
   Ztilde = (Xtilde(:,j)'*pinv(C_Theta_2_nor))' ;
   
     

      
    
    Recon_X(:,j) = (Ztilde'*Theta_2_nor)' + tempmean ; 
     


end 
    
    Sparse_Err = OriginX-Recon_X ;
    Sparse_MSE = sqrt(sum(sum(Sparse_Err.^2)))/(nrow2*M) ;
    



 MSE_Sparse_p(1,p/6)= Sparse_MSE ;

end

 MSE_Recon_ELM_Gram(k/6,:)= MSE_Sparse_p ;

end


    file = fopen('MSE_Exact_ELM_Gram.out', 'w');
    fprintf(file,'%s\r\n','MSE_Exact_ELM_Gram');
    fclose(file);           
    dlmwrite('MSE_Exact_ELM_Gram.out',MSE_Exact_ELM_Gram,'delimiter' , ',' , ...
          'precision','%30.16f','-append','newline','pc');
      
    file = fopen('MSE_Recon_ELM_Gram.out', 'w');
    fprintf(file,'%s\r\n','MSE_Recon_ELM_Gram');
    fclose(file);           
    dlmwrite('MSE_Recon_ELM_Gram.out',MSE_Recon_ELM_Gram,'delimiter' , ',' , ...
          'precision','%30.16f','-append','newline','pc');

       
       