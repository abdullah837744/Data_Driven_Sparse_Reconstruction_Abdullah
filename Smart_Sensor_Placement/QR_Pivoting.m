%%%%%% QR Pivoting by Abdullah %%%%%

function [indices]=Pivot_sensor(X,p)

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
phi = Xhalf*Vsort(:,1:cut)*diag(1./S);


[Q,R,pivot] = qr(phi*phi','vector') ;

indices = pivot(1:p) ;

end
