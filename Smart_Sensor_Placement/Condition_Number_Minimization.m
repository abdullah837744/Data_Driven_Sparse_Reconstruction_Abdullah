%%%%%%%%% Condition Number Minimization by Abdullah %%%%%%%%%%


function [ind_vec]= Cond_sensor(X,p)

nrow = (size(X,1))/2 ;
Xhalf = X;
Xhalf(nrow+1:end,:) = [];
[n m] = size(Xhalf) ;


R = Xhalf'*Xhalf;
[V,D] = eig(R);
for i = 1:m
    if D(i,i) < 0
        D(i,i) = 0;
    end
end
[~,I] = sort(diag(D),'descend');
DD=diag(D);
Dsort=diag(DD(I));
Vsort=V(:,I);
S = sqrt(diag(Dsort));
S(find(S<1e-12))=[];
cut = size(S,1);
phi_full = Xhalf*Vsort(:,1:cut)*diag(1./S);

for j = 1:p

    remain = linspace(1,n,n) ;
    if j > 1
        remain(ind_vec) = [];
    end



for i = 1:n+1-j

    location = j
    iteration = i

    remain0 = remain ;
    remain0(i) = [];
    phi = phi_full ;
    phi(remain0,:) = 0;

M = phi'*phi ;
cond_vec(i) = cond(M) ;

end
ind = find(cond_vec == (min(cond_vec))) ;
ind_vec(j) = remain(ind(1));

end

end
