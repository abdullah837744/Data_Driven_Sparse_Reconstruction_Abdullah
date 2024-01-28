%%%%%%% Discrete Impirical Interpolation Method by Abdullah %%%%%%%%%

function [indices]=DEIM(X)

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

indices = zeros(cut,1);

[mx in] = max(abs(phi(:,1)));
indices(1)= in ;
p = zeros(n,1);
p(in) = 1;

for j = 2:cut

    a = (pinv(p'*phi(:,1:j-1)))*(p'*phi(:,j)) ;

    r = phi(:,j) - (phi(:,1:j-1)*a) ;

    [mx in] = max(abs(r));
    indices(j)= in ;
    p_ad = zeros(n,1);
    p_ad(in) = 1;

    p = [p p_ad] ;

end

end
