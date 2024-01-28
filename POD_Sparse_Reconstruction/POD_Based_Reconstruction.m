%%%%%%% POD based Reconstruction by Abdullah %%%%%%%%%

clc
clear all
%%%%%%%POD Analysis %%%%%%%%%
F_Platform      = 2 ;

	if (F_Platform == 1 )
        cd ('C:\Users\Abdullah\Desktop\ELM_POD\POD Results')
        addpath ('D:\Chen LU\SparReconTestCase\Library')
        addpath ('D:\Chen LU\Unknown Basis\Reduced Channel\PODcutoff - 50\Input')
	elseif (F_Platform == 2 )
		cd ('/home/cfsr123/Abdullah/Blend_Manuscript_Chen/Plots/Page-25/Results')
		addpath ('/home/cfsr123/Abdullah/Library')
		addpath ('/home/cfsr123/Abdullah/Cylinder_Re_100')
	end





    fstart        = 1501               ;       % starting file  number
    fend          = 1800             ;       % ending file  number
    finc          = 1               ;       % file increment
    iname         = 'Input_'         ;       % input file name
    iext          = '.in'           ;       % input file name extension
    lowertest     = 1               ;       % Repairing starting snapshot
	uppertest     = 300             ;       % Repairing ending snapshot
	testinc       = 1               ;       % Repairing snapshot increment


    M = 0                           ;
    delimiterIn   = ''              ;       % Delimiter of the data
    headerlinesIn = 2               ;       % Headerline number of the data file
    columnheader  = 0               ;       % Columnline number of the data file
    scalar        = 2               ;       % 1 is u, 2 is u & v, 3 is u,v,w
    dimen         = 2               ;       % 1 is x, 2 is x & y, 3 is x,y,z
    PODMean       = 0               ;       % 1 is to take mean out during POD procedure, 0 is not
    PODTruncation = 1               ;       % 1 is rank based cutoff; 0 is % based cutoff
    PODoutput     = 0               ;       % Output POD modes or not, 0 no; 1 yes

    k95           = 2             ;
    points        = 20               ;       % Number of measurement points
    select_k      = 2:2:6              ;       % number of selected mode number

    Reg_Parameter = 1e-6               ;  % Regularization parameter when P < K

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

    MSE_Exact_Recon = zeros(length(select_k),2);
    MSE_Sparse_Recon = zeros(length(select_k),length(points));







    for k =select_k

    mode = k/k95;
    disp(['Computing for mode: ', int2str(mode)]);
    %%Exact Recon
    Exact_Recon=zeros(nrow2,M);
    [phi,~,~,nmode,~]=Compute_PODMode(X,PODMean,k,PODTruncation,PODoutput,rawdata(:,1:2));
    ProjectedA=phi'*X;

     file = fopen(['FR_Coeff_mat' num2str(k) '.out'], 'w');
     fprintf(file,'%s\r\n','FR_Coeff_mat' );
     fclose(file);
     dlmwrite(['FR_Coeff_mat' num2str(k) '.out'],ProjectedA,'delimiter' , ',' , ...
           'precision','%30.16f','-append','newline','pc');




    for i=1:M
        Exact_Recon(:,i)=phi*ProjectedA(:,i)+tempmean;

    end

    Exact_Err = OriginX-Exact_Recon ;
    Exact_MSE = sqrt(sum(sum(Exact_Err.^2)))/(nrow2*M) ;
    MSE_Exact_Recon(k/k95,1)= k;
    MSE_Exact_Recon(k/k95,2)= Exact_MSE;


    %% sparse Recon
    MSE_Sparse_p = zeros(1,length(points));

    for p = points

    pnt = p/k95;
    disp(['Computing for point: ', int2str(pnt)]);

    for testnum=lowertest:testinc:uppertest
%     disp(['Reconstructing Snapshot: ', int2str(testnum)]);

    meas = p;        % Number of measurements
    unmeas=nrow-meas;

    rng('default');
    rng(101)
    randnum = randperm(nrow)             ;
    index   = randnum(1:meas)            ;
    index2  = randnum(meas+1:nrow)       ;

    mask       = zeros(nrow*2,1) ;
    mask(index)= 1               ;
    mask(index+nrow)=1           ;

    NewX = OriginX(:,testnum)- tempmean ;

    maskX =mask.*NewX;


    %Construct M matrix
     Mmatrix=zeros(nmode,nmode);
          for j=1:nmode
                for i=1:nmode
                        Mmatrix(i,j)=(mask(:).*phi(:,j))'*(mask(:).*phi(:,i));
                 end
          end

     %Construct f matrix
      fmatrix=zeros(nmode,1);
           for i= 1:nmode
                 fmatrix(i)=(mask.*phi(:,i))'*maskX;
           end

       if meas > nmode
            coeff=Mmatrix\fmatrix;
       else
	     [~,s,~]=svd(Mmatrix,'econ');
	     coeff=(Mmatrix+max(diag(s))*Reg_Parameter*eye(nmode))\fmatrix;
       end
		Xprime=phi*coeff;

        X_sparse_recon (:,testnum) = Xprime + tempmean ;
        Coeff_mat (:,testnum) = coeff ;

       end

     %[~]=Plot_Field1(rawdata(:,1),rawdata(:,2),X_sparse_recon(1:nrow,1),'Recon-Re100-K= ',k,0,2)
     [~]=Plot_FieldLine2(rawdata(:,1),rawdata(:,2),OriginX(1:nrow,1),X_sparse_recon(1:nrow,1),'Recon-Re100(line)-K= ',k,0,1)

    file = fopen(['SR_Coeff_mat' num2str(k) '_' num2str(p) '.out'], 'w');
     fprintf(file,'%s\r\n','SR_Coeff_mat' );
     fclose(file);
     dlmwrite(['SR_Coeff_mat' num2str(k) '_' num2str(p) '.out'],Coeff_mat,'delimiter' , ',' , ...
           'precision','%30.16f','-append','newline','pc');

    clear Coeff_mat



    Sparse_Err = OriginX-X_sparse_recon ;
    Sparse_MSE = sqrt(sum(sum(Sparse_Err.^2)))/(nrow2*M) ;
    MSE_Sparse_p(1,1)= Sparse_MSE ;

    end

    MSE_Sparse_Recon(k/k95,:) = MSE_Sparse_p;

    end

ep1 = MSE_Sparse_Recon / MSE_Exact_Recon(1,2) ;
ep2 = MSE_Sparse_Recon./ MSE_Exact_Recon(:,2) ;


     file = fopen('ep1.out', 'w');
    fprintf(file,'%s\r\n','ep1');
    fclose(file);
    dlmwrite('ep1.out',ep1,'delimiter' , ',' , ...
          'precision','%30.16f','-append','newline','pc');

    file = fopen('ep2.out', 'w');
    fprintf(file,'%s\r\n','ep2.out');
    fclose(file);
    dlmwrite('ep2.out',ep2,'delimiter' , ',' , ...
          'precision','%30.16f','-append','newline','pc');


    file = fopen('MSE_Exact_Recon.out', 'w');
    fprintf(file,'%s\r\n','MSE_Exact_Recon');
    fclose(file);
    dlmwrite('MSE_Exact_Recon.out',MSE_Exact_Recon,'delimiter' , ',' , ...
          'precision','%30.16f','-append','newline','pc');

    file = fopen('MSE_Sparse_Recon.out', 'w');
    fprintf(file,'%s\r\n','MSE_Sparse_Recon.out');
    fclose(file);
    dlmwrite('MSE_Sparse_Recon.out',MSE_Sparse_Recon,'delimiter' , ',' , ...
          'precision','%30.16f','-append','newline','pc');
