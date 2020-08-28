import torch 


@torch.no_grad()
def CG(A_bmm, b, optTol, verbose, maxIter):
    x = torch.zeros(b.size()[0]).cuda()
    r = -b
    rr = r.t().dot(r)
    p = -r
    k = 0
    res = torch.norm(r)
    done = 0 

    while res > optTol and k < maxIter and not done:
        Ap = A_bmm(p)
        pAp = p.t().dot(Ap)

        if pAp <= 1e-16:
            print(k, pAp.item(), "Non-positive curvature detected")

            if pAp < 0:
                # negative 
                print(k, pAp.item(),  "Negative curvature detected")
                p = p / torch.norm(p)
                p = p / torch.sum(torch.abs(b))

                return p
            
            if k == 0:
                print("first iter, proceeding")
                done = 1 
            else:
                print("Stopping")
                break
        else:
            # conjugate gradient
            alpha = rr / pAp
            x = x + alpha * p
            r = r + alpha * Ap

            rr_new = r.t().dot(r)
            beta = rr_new / rr 
            p = -r + beta * p 
            
            # update variables
            rr = rr_new 
            res = torch.norm(r)
            k +=1 


    return x