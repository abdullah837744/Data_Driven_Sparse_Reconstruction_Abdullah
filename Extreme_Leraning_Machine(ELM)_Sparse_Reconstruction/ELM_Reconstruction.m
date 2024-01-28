%%%%%%%%%%%% Extreme_Learning_Machine by Abdullah %%%%%%%%%%%
F_Platform      = 2 ;

	if (F_Platform == 1 )
        cd ('C:\Users\Abdullah\Desktop\Fao')
        addpath ('D:\Chen LU\SparReconTestCase\ChenLuLibrary')
        addpath ('D:\Chen LU\Hard Disk\SimulationData\CylinderData\deltaT=0.2s\Re100t')
        addpath ('D:\Chen LU\Abdullah\ELM AutoEncoder\ELM_Library')



	elseif (F_Platform == 2 )
		cd ('/home/cfsr123/Abdullah/ELM_Exact_Sparse_Recon/Results')
		addpath ('/home/cfsr123/Abdullah/ELM_Library')
		addpath ('/home/cfsr123/Abdullah/Input')
        addpath ('/home/cfsr123/Abdullah/ChenLuLibrary')
	end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    fstart        = 1500               ;       % starting file  number
    fend          = 1800             ;       % ending file  number
    finc          = 1               ;       % file increment
    iname         = 'Input_'         ;       % input file name
    iext          = '.in'           ;       % input file name extension
    lowertest     = 1               ;       % Repairing starting snapshot
	uppertest     = 301             ;       % Repairing ending snapshot
	testinc       = 1               ;       % Repairing snapshot increment



    H95           =98              ;
    M = 0                           ;
    delimiterIn   = ''              ;       % Delimiter of the data
    headerlinesIn = 2               ;       % Headerline number of the data file
    columnheader  = 0               ;       % Columnline number of the data file
    scalar        = 2               ;       % 1 is u, 2 is u & v, 3 is u,v,w
    dimen         = 2               ;       % 1 is x, 2 is x & y, 3 is x,y,z

    rand_seed = 101 ;

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
    %Num_H = round(nrow2/100);

    	%Take the mean out
	tempmean=mean(X,2);
	OriginX=X;
	for i=1:M
		X(:,i)=X(:,i)-tempmean;
    end


%Read_Num_H = dlmread('x_ELM_predict.out',',',1,0);
Num_H = 98;
points = 98;

    MSE_Exact_Recon_ELM = zeros(length(Num_H),2);
    MSE_Sparse_Recon_ELM = zeros(length(Num_H),length(points));

    for mask_seed = 101:110

        mask_seed

for k = Num_H

basis = k/H95;
disp(['Computing for Select Hidden Layer: ', int2str(basis)]);

A = [X' X'];

 [Train_Time,Train_Accuracy,Theta_1,Theta_2,Z,Predicted_X] = ELM_seed_theta1(A,0,k,'radbas',nrow2,rand_seed);

    Exact_Err = X-Predicted_X ;
    Exact_MSE = sqrt(sum(sum(Exact_Err.^2)))/(nrow2*M) ;
    MSE_Exact_Recon_ELM(k/H95,1)= k;
    MSE_Exact_Recon_ELM(k/H95,2)= Exact_MSE;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     ptTheta_2 = pinv(Theta_2');
%     tTheta_2 = Theta_2';
%
%     I = zeros(nrow2,1);
%
%     for i=1:nrow2
%         I(i,1)= tTheta_2(i,:)*ptTheta_2(:,i);
%     end
%
%
%     Iden = ones(nrow2,1);
%
%     I_err = I-Iden ;
%
%     I_err_MSE = sqrt(sum(sum(I_err.^2)))/(nrow2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MSE_Sparse_p = zeros(1,length(points));
for p = points
    pnt = p/H95;
    disp(['Computing for point: ', int2str(pnt)]);

%% Construct mask vector
    meas = p;        % Number of measurements
    unmeas=nrow-meas;



    rng('default');
    rng(mask_seed)
    randnum = randperm(nrow)             ;
    index   = randnum(1:meas)            ;
    index2  = randnum(meas+1:nrow)       ;

    mask       = zeros(nrow*2,M) ;
    NewX       = zeros(nrow*2,M) ;

    for j = 1:M
    mask(index,j)= 1               ;
    mask(index+nrow,j)=1           ;
    NewX(:,j) = OriginX(:,j)- tempmean ;
    end

    Xtilde = mask.*NewX;



%% Reconstructing
C = mask';


Recon_X = zeros(nrow2,M);
Pred_X = zeros(nrow2,M);



MSE = 0 ;
for j = 1:M


    for i = 1:k
        C_Theta_2(i,:)=C(j,:).*Theta_2(i,:);

    end

     Ztilde = (Xtilde(:,j)'*pinv(C_Theta_2))' ;





      Recon_X(:,j) = (Ztilde'*Theta_2)' + tempmean ;

%      Pred_X(:,j) = (Z(:,j)'*Theta_2)' + tempmean ;

end

    Sparse_Err = OriginX-Recon_X ;
    Sparse_MSE = sqrt(sum(sum(Sparse_Err.^2)))/(nrow2*M) ;

%     Sparse_Err_pred = OriginX-Pred_X ;
%     Sparse_MSE_pred = sqrt(sum(sum(Sparse_Err_pred.^2)))/(nrow2*M);


 MSE_Sparse_p(1,p/H95)= Sparse_MSE ;

end

 MSE_Sparse_Recon_ELM(k/H95,:)= MSE_Sparse_p ;

end

mask_seed_err(mask_seed-100,1) = MSE_Sparse_Recon_ELM ;

   end

    file = fopen('MSE_Exact_Recon_ELM.out', 'w');
    fprintf(file,'%s\r\n','MSE_Exact_Recon_ELM');
    fclose(file);
    dlmwrite('MSE_Exact_Recon_ELM.out',MSE_Exact_Recon_ELM,'delimiter' , ',' , ...
          'precision','%30.16f','-append','newline','pc');

%     file = fopen('MSE_Sparse_Recon_ELM.out', 'w');
%     fprintf(file,'%s\r\n','MSE_Sparse_Recon_ELM.out');
%     fclose(file);
%     dlmwrite('MSE_Sparse_Recon_ELM.out',MSE_Sparse_Recon_ELM,'delimiter' , ',' , ...
%           'precision','%30.16f','-append','newline','pc');



        file = fopen('mask_seed_err.out', 'w');
    fprintf(file,'%s\r\n','mask_seed_err.out');
    fclose(file);
    dlmwrite('mask_seed_err.out',mask_seed_err,'delimiter' , ',' , ...
          'precision','%30.16f','-append','newline','pc');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CN = zeros(nrow2);
%
% for m = 1:nrow2
%
%     CN(m,m)= mask(m,1);
%
% end



% I = (Theta_2'*(pinv(C_Theta_2')))*CN ;
%
% Iden = eye(nrow2);
%
% I_err = I-Iden ;
%
% I_err_diag = diag(I_err);
%
% I_err_MSE_diag = sqrt(sum(sum(I_err_diag.^2)))/(nrow2*nrow2);


%     file = fopen('I_err_MSE_diag.out', 'w');
%     fprintf(file,'%s\r\n','I_err_MSE_diag.out');
%     fclose(file);
%     dlmwrite('I_err_MSE_diag.out',I_err_MSE_diag,'delimiter' , ',' , ...
%           'precision','%30.16f','-append','newline','pc');


%%%%%%%%%%%%%%%%%%%%%%%%%
% I2 = Theta_2'*(pinv(Theta_2'));
%
% I_err_2 = I2-Iden ;
%
% I_err_MSE_2 = sqrt(sum(sum(I_err_2.^2)))/(nrow2*nrow2);
%
%     file = fopen('I_err_MSE_2.out', 'w');
%     fprintf(file,'%s\r\n','I_err_MSE_2.out');
%     fclose(file);
%     dlmwrite('I_err_MSE_2.out',I_err_MSE_2,'delimiter' , ',' , ...
%           'precision','%30.16f','-append','newline','pc');
