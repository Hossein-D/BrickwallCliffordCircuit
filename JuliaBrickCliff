#Pkg.add("NPZ")

using Distributed
using Base.Threads
using SharedArrays
using Pkg
using Plots
using FileIO
using NPZ
using StatsBase
using VegaDatasets
using Distributed
using CSV, DataFrames, FileIO
using QuantumClifford
using JLD2
using QuantumClifford

file = jldopen("ru2.jld2", "r")
ru2 = file["ru2"]
close(file)

function returnPaul(p,sign=0x0)
    n=length(p)
    return PauliOperator(sign,convert.(Bool,p[1:n÷2]),convert.(Bool,p[n÷2+1:n]))
end

function RandUnitaries()
    n = 0
    Nline = 2880

    unitaries = zeros(2880, 4)
    x = [0:1:10;]
    open("twoUnitaries.txt") do f     
        line = 0   
        while ! eof(f) #&& line<10  
            s = readline(f)           
            line += 1
            unitaries[line, 1] = parse(Int64, s[1])
            unitaries[line, 2] = parse(Int64, s[3])
            unitaries[line, 3] = parse(Int64, s[5])
            unitaries[line, 4] = parse(Int64, s[7])
        
        end 
    end 
    
    randU = zeros(Nline, 4)
    n=0
    for n in 0:Int(Nline/4)-1
        randU[4*n+1, 1:4] = copy(unitaries[4*n+1:4*n+4, 3]) # Z1 line: 0th line of n'th randU
        randU[4*n+2, 1:4] = copy(unitaries[4*n+1:4*n+4, 1])# X1 line: 1st line of n'th randU
        randU[4*n+4, 1:4] = copy(unitaries[4*n+1:4*n+4, 4]) # Z2 line: 3rd line of n'th randU
        randU[4*n+3, 1:4] = copy(unitaries[4*n+1:4*n+4, 2]) # X2 line: 2nd line of n'th randU    
    end
    swap = zeros(Nline)
    swap[1:Nline] = randU[:, 1]
    randU[:, 1] = randU[:, 4]
    randU[1:Nline, 4] = swap[1:Nline]   
    return randU
end


function initProdState(N, ancilla)
    N = Int(N)
    if !Bool(ancilla)
        Ginit = zeros(Int8, 2*N, 2*N+1)
        for i in 1:N
            Ginit[i, 2*i-1] = 1    # Z stabilizers are Z operators
            Ginit[i+N, 2*i] = 1 # X stabilizers are X operators
        end

    else
        N = Int(N)
        Ginit = zeros(Int8, 2*N, 2*N+1)
        for i in 1:N
            Ginit[i, 2*i-1] = 1    # Z stabilizers are X operators
            Ginit[i+N, 2*i] = 1 # X stabilizers are Z operators
        end
    end    
        #print('Sinit Shape \n', Sinit.shape)
    return Ginit 
end


function initProdStateShort(N, ancilla)
    # Creates initial states
    # We remove the sign bit at the end of the Pauli operators.     
    N = Int(N)
    if !Bool(ancilla)
        Ginit = zeros(Int8, 2*N, 2*N+1)
        for i in 1:N
            Ginit[i, 2*i-1] = 1    # Z stabilizers are Z operators
        end
    else
        N = Int(N)
        Ginit = zeros(Int8, 2*N, 2*N+1)
        for i in 1:N
            Ginit[i, 2*i-1] = 1    # Z stabilizers are X operators
        end
    end    
    return Ginit 
end



function fourbitFunc(a, b, c, d)
    if a==0 && b==0
        return 0
    end
    if a==1 && b==1
        return c-d
    end
    if a==1 && b==0
        return d*(1-2*c)
    end
    if a==0 && b==1
        return c*(2*d-1)
    end    
end


function xor(v1, v2)
    # Evaluates the logical XOR of two vectors
    return [Int(sum(x)%2) for x in zip(v1, v2)]
end

function PauliProd(P1, P2)
    # Evaluates the product of two Pauli operators. 
    # Takes two vectorial Pauli Operators. 
    N = Int(round((length(P1)-1)/2)) 
    r1 = P1[2*N+1]
    r2 = P2[2*N+1]
    sumG = 0
    sumPauli = xor(P1[1:Int(2*N)], P2[1:Int(2*N)])
    for i in 1:N
        sumG += fourbitFunc(P1[2*i-1],P1[2*i],P2[2*i-1],P2[2*i])
    end
    return sumPauli, abs(((2*r1 + 2*r2 + sumG) % 4)/2)
end

function PauliProdShort(P1, P2)
    N = Int(round((length(P1)-1)/2))
    sumG = 0
    sumPauli = xor(P1[1:Int(2*N)], P2[1:Int(2*N)])    
    return sumPauli
end

function inputFunc(a, b, c=3)
    println("c = ", c)
    if isempty(c)
        println(a)
        println("c = isempty")
    end
end

function sumx1x2(x1, x2)
    return cx1, x1[1]+x2
end

function enlargeU_IZXY(U)    
    N = Int(round(size(U)[1]/2))
    g = zeros(N, 2, 2, 2*N+1)
    for i in 1:Int(N)
        g[i, 2, 1, 1:2*N+1] = copy(U[2*i, 1:2*N+1])
        g[i, 1, 2, 1:2*N+1] = copy(U[2*i-1, 1:2*N+1])
        g[i, 2, 2, 1:2*N] = copy(xor(U[2*i-1, 1:2*N], U[2*i, 1:2*N]))
        prodUxUz, prodSignXSignZ = PauliProd(U[2*i-1, :], U[2*i, :])
        g[i, 2, 2, 2*N+1] = ((1 + 2*prodSignXSignZ) % 4)/2
    end
    return g
end


function PauliTransform(P, U)
    # Applies a unitary operator to a Pauli P.
    N = Int(round(size(U)[1]/2))
    numRows = size(P)[1]
    newP = zeros(size(P))
    for r in 1:numRows
        PBit = P[r, 2*N+1]
        transformP = transpose(P[r, 1:2*N])*U.%2
        enlargeU = enlargeU_IZXY(U)
        SBitProd = 0
        sumPauli = zeros(1, 2*N+1)
        newP[r, 1:2*N] = transformP[1:2*N]
        SBitProd = 0
        for n in 1:N-1
            tempSumPauli = PauliProd(enlargeU[n, Int(P[r, n])+1, Int(P[r, n+1])+1, 1:2*N+1], sumPauli)
            SBitProd = tempSumPauli[2]
            sumPauli[1, 1:2*N] = tempSumPauli[1][1:2*N]
            sumPauli[1, 2*N+1] = tempSumPauli[2]
        end
        transformPBit = (PBit + SBitProd)%2
        newP[r, 1:2*N] = copy(transformP[1:2*N])
        newP[r, 2*N+1] = copy(transformPBit)
    end        
    return newP
end



function PauliTransformShort(P, U)
    # Applies a unitary operator to a Pauli P without considering the sign bit. 
    N = Int(round(size(U)[1]/2))
    numRows = size(P)[1]
    newP = zeros(size(P))
    for r in 1:numRows        
        transformP = transpose(P[r, 1:2*N])*U.%2
        newP[r, 1:2*N] = copy(transformP[1:2*N])
    end    
    return newP
    
end



function symplecticProd(P1, P2)
    # Evaluates the symplectic product of two Pauli operators. 
    N = Int((length(P1)-1)/2)
    sumP = 0
    for i in 1:N
        sumP = (sumP + P1[2*i]*P2[2*i-1] + P1[2*i-1]*P2[2*i])
    end
    return sumP%2
end



function symplecticProdShort(P1, P2)
    # Evaluates the symplectic product of two Pauli operators without considering the sign bit.  
    N = Int((length(P1)-1)/2)
    sumP = 0
    for i in 1:N
        sumP = (sumP + P1[2*i]*P2[2*i-1] + P1[2*i-1]*P2[2*i])
    end
    return sumP%2
end


function measurePauli(g, S)
    # Measures an operator g on a state stabilized with stabilizer S. 
    N = Int((size(g)[2]-1)/2)
    orthoSymProd = zeros(N)
    ZantiCommute = zeros(size(g))
    XantiCommute = zeros(size(g))   
    Zindex = Int64[]
    Xindex = Int64[]
    Zcounter = 0
    Xcounter = 0
    newSZindex1 = zeros(size(g))
    newS = zeros(Float64, size(S))
    newStabX = zeros(1, 2*N+1)
    newStabZ = zeros(1, 2*N+1)
    
    measureRes = NaN  # "measureRes = x" means measure_Result = i^{2x}
    deterministic = 0
    for i in 1:N
        Zprod = symplecticProd(g, S[i,:])
        if Zprod == 1
            Zcounter += 1
            append!(Zindex, Int(i))
            if Zcounter == 1
                ZantiCommute[1, :] = copy(S[i,:])
            else
                ZantiCommute = vcat(ZantiCommute, S[i,:]')
            end
        end
        Xprod = symplecticProd(g, S[i+N,:])
        if Xprod == 1    
            Xcounter += 1
            append!(Xindex, Int(i+N))
            if Xcounter == 1
                XantiCommute[1, :] = copy(S[i+N,:])
            else
                XantiCommute = vcat(XantiCommute, S[i+N,:]')                
            end
        end
    end
    newS = copy(S)
    if Zcounter == 0
        deterministic = 1
        plusMinusG = zeros(Int(2*N)+1)
        for i in 1:N
            coefX = Int(symplecticProd(g, S[i+N, :]))
            tempPlusMinusGProd = PauliProd(coefX*S[i, :], plusMinusG)
            tempPlusMinusG = xor(coefX*S[i, 1:2*N], plusMinusG)
            plusMinusG[1:2*N] = copy(tempPlusMinusG)
            plusMinusG[2*N+1] = copy(tempPlusMinusGProd[2])
        end
        for i in 1:2*N 
            if plusMinusG[i] != g[i]
            end
        end
        if plusMinusG[2*N+1] == g[2*N+1]
            measureRes = 0
        else
            measureRes = (g[2*N+1]-plusMinusG[2*N+1])%2
        end    
        if measureRes < 0
            measureRes += 2
        end
        for n in 1:N
            orthoSymProd[n] = symplecticProd(newS[n, :], newS[n+Int(N), :])
        end
    elseif Zcounter >= 1
        for i in 2:Zcounter
            tempProdZ = PauliProd(S[Zindex[i], :], S[Zindex[1], :])
            newStabZ[1, 1:2*N] = copy(tempProdZ[1][1:2*N]) 
            newStabZ[1, 2*N+1] = copy(tempProdZ[2])
            newS[Zindex[i], :] = copy(newStabZ[:])
        end
        for i in 1:Xcounter
            tempProdX = PauliProd(S[Int(Xindex[i]), :], S[Int(Zindex[1]), :])
            newStabX[1, 1:2*N] = copy(tempProdX[1][1:2*N])
            newStabX[1, 2*N+1] = copy(tempProdX[2])
            newS[Xindex[i], 1:2*N+1] = copy(newStabX[1, 1:2*N+1])
        end
        for m in 1:N
            orthoSymProd[m] = symplecticProd(newS[m, :], newS[m+Int(N), :])
        end
        newSZindex1[:] = copy(newS[Int(Zindex[1]), :])
        newS[Int(Zindex[1]), :] = copy(g)        
        randVec = rand(Float64)        
        coin = [Bool(i<0.5) for i in randVec]   
        if Bool(coin[1])
            newS[Int(Zindex[1]), Int(2*N)+1] = (g[Int(2*N)+1]+1)%2  #Int((g[Int(2*N)+1]+1)%2)  
            measureRes = 1 # measureRes = x means i^{2x}; x = 1 -> m=-1
        else
            measureRes = 0 # measureRes = x means i^{2x}; x=0 -> m=0
        end
        newS[Int(Zindex[1])+N, :] = copy(newSZindex1[:])            
    end
    tempMeasure = 0
    if measureRes == 0
        tempMeasure = 1 #S_z = -1
    elseif isnan(measureRes)
        tempMeasure = 0     # No measurement           
    elseif measureRes == 1.0
        tempMeasure = 2  # S_z = +1
    elseif measureRes == 1.5
        tempMeasure = 1.5  # S_z = +1
    elseif measureRes == 0.5
        tempMeasure = 0.5  # S_z = +1        
    end
    measureRes = tempMeasure    
    return newS, measureRes, deterministic #ZantiCommute, XantiCommute
end

function measurePauliShort(g, S)
    N = Int((size(g)[2]-1)/2)
    orthoSymProd = zeros(N)
    ZantiCommute = zeros(size(g))
    XantiCommute = zeros(size(g))   
    Zindex = Int64[]
    Xindex = Int64[]
    Zcounter = 0
    Xcounter = 0    
    newSZindex1 = zeros(size(g))
    newS = zeros(Float64, size(S))
    #newS[1, 1] = 1.4
    newStabX = zeros(1, 2*N+1)
    newStabZ = zeros(1, 2*N+1)    
    measureRes = NaN  # "measureRes = x" means measure_Result = i^{2x}
    deterministic = 0    
    for i in 1:N        
        Zprod = symplecticProd(g, S[i,:])
        if Zprod == 1
            Zcounter += 1            
            append!(Zindex, Int(i))
            if Zcounter == 1
                ZantiCommute[1, :] = copy(S[i,:])                
            else
                ZantiCommute = vcat(ZantiCommute, S[i,:]')
            end
        end        
        Xprod = symplecticProd(g, S[i+N,:])
        if Xprod == 1
            Xcounter += 1
            append!(Xindex, Int(i+N))
            
            if Xcounter == 1
                XantiCommute[1, :] = copy(S[i+N,:])
            else
                XantiCommute = vcat(XantiCommute, S[i+N,:]')                                
            end
        end
    end
    newS = copy(S)
    if Zcounter == 0
        deterministic = 1
        plusMinusG = zeros(Int(2*N)+1)
        for i in 1:N
            coefX = Int(symplecticProd(g, S[i+N, :]))            
            tempPlusMinusGProd = PauliProd(coefX*S[i, :], plusMinusG)
            tempPlusMinusG = xor(coefX*S[i, 1:2*N], plusMinusG)
            plusMinusG[1:2*N] = copy(tempPlusMinusG)
            plusMinusG[2*N+1] = copy(tempPlusMinusGProd[2])
        end
        for i in 1:2*N 
            if plusMinusG[i] != g[i]
                #println("not equal")
                #exit()
            end
        end
    elseif Zcounter >= 1
        for i in 2:Zcounter
            tempProdZ = PauliProd(S[Zindex[i], :], S[Zindex[1], :])            
            newStabZ[1, 1:2*N] = copy(tempProdZ[1][1:2*N]) 
            newStabZ[1, 2*N+1] = copy(tempProdZ[2])
            newS[Zindex[i], :] = copy(newStabZ[:])
        end
        for i in 1:Xcounter
            tempProdX = PauliProd(S[Int(Xindex[i]), :], S[Int(Zindex[1]), :])
            newStabX[1, 1:2*N] = copy(tempProdX[1][1:2*N])
            newStabX[1, 2*N+1] = copy(tempProdX[2])            
            newS[Xindex[i], 1:2*N+1] = copy(newStabX[1, 1:2*N+1])            
        end
        for m in 1:N            
            orthoSymProd[m] = symplecticProd(newS[m, :], newS[m+Int(N), :])
        end
        newSZindex1[:] = copy(newS[Int(Zindex[1]), :])
        newS[Int(Zindex[1]), :] = copy(g)        
        randVec = rand(Float64)        
        coin = [Bool(i<0.5) for i in randVec]           
        if Bool(coin[1])
            newS[Int(Zindex[1]), Int(2*N)+1] = (g[Int(2*N)+1]+1)%2  #Int((g[Int(2*N)+1]+1)%2)  
        end
        newS[Int(Zindex[1])+N, :] = copy(newSZindex1[:])            
    end
    return newS, deterministic    
end

function projectOut(Matrix, RegionA)
    # Projects out the wave function on a region A. 
    # RegionA: The first element is the starting point of region A, 
    # and the second component is the length of region A
    # sizeA = size(RegA)[1]
    numRow = size(Matrix)[1]
    projectA = zeros(Int(numRow), Int(RegionA[2]))
    projectA[1:numRow, 1:RegionA[2]] = copy(Matrix[1:numRow, RegionA[1]:RegionA[1]+RegionA[2]-1])
    return projectA
end

function clippingGauge(A)
    m = size(A)[1]
    n = size(A)[2]
    h = 1
    k = 1
    tempArray = zeros(1, n)
    while h <= m && k <= n
        i_max = h-1+findmax(A[h:m, k])[2]
        if A[i_max, k] == 0
            k += 1
            continue            
        else
            tempArray[1, :] = A[h, :]
            A[h, :] = A[i_max, :]
            A[i_max, :] = tempArray[1, :]
        end
        for i in h+1:m
            if A[h, k]==0
                println("A[h, k]==0 \n", A)
            end
            f = A[i, k]/A[h, k]
            A[i, k] = 0
            for j in k+1:n
                A[i,j] = abs((A[i,j] - A[h,j]*f)%2)
            end
        end
        h += 1
        k += 1
    end
    echelonA = zeros(m, n)
    for i in 1:m
        echelonA[i, :] = copy(A[m-i+1, :])
    end
    
    zeroRow = 0
    for i in 1:size(echelonA)[1]
        for j in 1:size(echelonA)[2]
            #println('i, j', i, j)
            if (echelonA[i, j]==0) && j != size(echelonA)[2]
                continue
            elseif (echelonA[i, j]==0) && j == size(echelonA)[2]
                zeroRow += 1
                break
            else
                break 
            end
        end
    end
    rank = size(echelonA)[1] - zeroRow      
    return rank #, echelonA
end

function Entropy(M, A)
    # Evaluates the entanglement entropy of region A of a state stabilizer with M.
    lengthA = A[2]
    projA = projectOut(M, A)
    rankProjA = clippingGauge(projA)
    SA = rankProjA - lengthA/2    
    return SA
end

function checkOrtho(G)
    # Checks the orthogonality of a given stabilizer. 
    N = Int(size(G)[1]/2)
    ortho = zeros(2*N, 2*N)
    for i in 1:2*N
        for j in 1:2*N
            ortho[i, j] = symplecticProd(G[i, :], G[j, :])
        end
    end    
    return ortho
end   


function convertStabRep(S)
    # Converts the representation of a stabilzier S. 
    N = Int((size(S)[2]-1)/2)
    signVec = zeros(UInt8, N)
    for i in 1:N
        signVec[i] = 0x0
    end
    Xbits = zeros(Bool, N, N)
    Zbits = zeros(Bool, N, N)    
    signI = 0x0
    for i in 1:N
        if S[i, 2*N+1]==0
            signI = 0x0            
        elseif S[i, 2*N+1]==0.5
            signI = 0x1
        elseif S[i, 2*N+1]==1
            signI = 0x2            
        elseif S[i, 2*N+1]==1.5
            signI = 0x3            
        end
        signVec[i] = signI
        for j in 1:N
            Xbits[i, j] = Bool(S[i, 2+2*(j-1)])    
            Zbits[i, j] = Bool(S[i, 1+2*(j-1)])
        end
    end
    stab = Stabilizer(signVec,Xbits,Zbits)
    return stab
end

function convBackStab(S)
    # Converts back the representation of a stabilzier S.     
    S_XZRep = stab_to_gf2(S)
    size1 = Int(size(S_XZRep)[1])
    size2 = size(S_XZRep)[2]
    SPhase = S.phases
    myRep = zeros(size1, size2+1)
    Xbits = zeros(Bool, size1, Int(size2/2))
    Zbits = zeros(Bool, size1, Int(size2/2))
    for i in 1:size1
        Xbits[i, :] = S_XZRep[i, 1:Int(size2/2)]
        Zbits[i, :] = S_XZRep[i, Int(size2/2)+1:size2]    
        myRep[i, 1:2:size2] = Zbits[i, :]
        myRep[i, 2:2:size2] = Xbits[i, :]        
        sign = 0
        if SPhase[i]==0x0
            sign = 0
        elseif SPhase[i]==0x1
            sign = 0.5
        elseif SPhase[i]==0x2            
            sign = 1
        elseif SPhase[i]==0x3
            sign = 1.5          
        end
        myRep[i, size2+1] = sign
    end   
    return myRep
    println("myRep = ", myRep)
end

function convertPauliRep(P)
    # Converts the representation of a Pauli.     
    N = Int((size(P)[2]-1)/2)
    Xbits = zeros(Bool, N)
    Zbits = zeros(Bool, N)
    
    sign = 0x0
    if P[1, 2*N+1]==0
        sign = 0x0            
    elseif P[1, 2*N+1]==0.5
        sign = 0x1
    elseif P[1, 2*N+1]==1
        sign = 0x2            
    elseif P[1, 2*N+1]==1.5
        sign = 0x3            
    end
    for j in 1:N
        Xbits[j] = Bool(P[1, 2+2*(j-1)])    
        Zbits[j] = Bool(P[1, 1+2*(j-1)])
    end
    Pauli = PauliOperator(sign,Xbits,Zbits)
    return Pauli
end

function QCliffMeasure(S, P, keepRes=true)
    # Measures a Pauli P on a state stabilized S using the QuantumClifford Package.
    if keepRes==true
        newstate, anticomindex, result = project!(S, P)     
    elseif keepRes==false 
        newstate, anticomindex = project!(S, P, keep_result=false)                 
    end
    deterministic = 0
    if anticomindex==0
        deterministic = 1
    else
        deterministic = 0
    end    
    if keepRes==true
        if isnothing(result)
            measureRes = rand([0x0,0x2])
            newstate.phases[anticomindex] = measureRes
            result = measureRes
        end
        
        if result == 0x0
            measureRes = 1
        elseif result == 0x1
            measureRes = 0.5
        elseif result == 0x2
            measureRes = 2
        elseif result == 0x3
            measureRes = 1.5        
        end            
        return newstate, measureRes, deterministic
    elseif keepRes==false
        return newstate, deterministic
    end
end




function convQCliffMeasure(S, P, keepRes=true)
    # Measures a Pauli P on a state stabilized S using the QuantumClifford Package.
    tempS = copy(S)
    PPauli = convertPauliRep(P)
    #println("PPauli = ", PPauli)
    convS  = convertStabRep(tempS)
    #println("convS = ", convS)    
    if keepRes==true
        measurePauliRes = QCliffMeasure(convS, PPauli)
    else
        measurePauliRes = QCliffMeasure(convS, PPauli, keepRes)
    end
    newS = convBackStab(measurePauliRes[1])
    if keepRes==true       
        measureRes = measurePauliRes[2]
        deterministic = measurePauliRes[3]
        return newS, measureRes, deterministic
    elseif keepRes==false
        deterministic = measurePauliRes[2]        
        return newS, deterministic    
    end    

end


function QCliffTimeEvolveAncilla(L, p, T, withScramble, randIntVec1 = [], randIntVec2 = [], probArray = [])
    # We have 2*L+1 qubits with the same number of Stabilizers. Of these number of qubits, one is an ancilla qubit which is located in the middle of the system's qubits. 
    # Time evolution is only applied to all the qubits except the ancilla qubit. 
    # 
    # print('probArray \n', probArray)
    # size(Ginit) = (4*L+2)*(4L+2)
    # The ancilla qubit is put at the end of the string of the qubits. 
    
    # We first time evolve a Clifford circuit without performing any measurements. Next, we make a measurement 
    # on the middle qubit and next, we maximally entangle the middle qubit of the stabilizer state with a 
    # reference qubit by forming a Bell pair. 
    
    T1 = 4*L # Time evolved to create a randomly entangled volume state. 
    T2 = T # Time to evolve the circuit after entangling the middle qubit with the reference qubit. 
    Ginit = zeros(2*L+1, 4*L+3)
    ancillaQbit = 4*L
    middleQbit = 2*L
    ancilla = 0
    initStabDestab = initProdState(2*L+1, ancilla) # Initial state is a product state along the X axis
    
    Ginit = initStabDestab[1:2*L+1, :] # Initial state is a product state along the X axis    
    
    Ginit[L, ancillaQbit+2] = 0   # X=0
    Ginit[L, ancillaQbit+1] = 1   # Z=1    
    
    Ginit[2*L+1, ancillaQbit+2] = 1 # X=1
    Ginit[2*L+1, ancillaQbit+1] = 0 # Z=0
    Ginit[2*L+1, middleQbit-1] = 0  # Z=0
    Ginit[2*L+1, middleQbit] = 1  # X=1
        
    GStab = zeros(size(Ginit))
    GStab = copy(Ginit)
    
    if size(probArray)[1] == 0
        givenRand = false
    else
        givenRand = true
    end

    if size(randIntVec1)[1] == 0
        givenU = false
    else
        givenU = true
    end
    
    GStabLastGate = zeros(2*L+1, 4)
    measureResVec = zeros(T, 2*L+1)
    measureResVec[:, :] .= NaN

    ancillMeasure = zeros(1, 4*L+3)
    ancillMeasure[1, 4*L+1] = 1      # Measuring vector index
    ancilMeasPauli = convertPauliRep(ancillMeasure)
    #println(ancilMeasPauli)
    ancillaVecX = zeros(1, 4*L+3)
    ancillaVecX[4*L+2] = 1 
    ancilVecXPauli = convertPauliRep(ancillaVecX)
    
    ancillaVecY = zeros(1, 4*L+3)
    ancillaVecY[4*L+2] = 1 
    ancillaVecY[4*L+1] = 1 
    ancilVecYPauli = convertPauliRep(ancillaVecY)        
    ancillaVecZ = zeros(1, 4*L+3)
    ancillaVecZ[4*L+1] = 1   
    ancilVecZPauli = convertPauliRep(ancillaVecZ)    
    
    # Time evolution WITHOUT measurement #
    ######################################
        
    # We start by applying unitary time evolution without any measurements. 
    pop = [0, 1]
    weights = [1-p, p]
    
    randBit = [0, 0, 0, 0]    
    A = [1, 4*Int(L)]
    randU = RandUnitaries()
    TScramble = Int(4*L); 
    if withScramble
        for t in 1:TScramble
            if !(givenRand)
                B = rand(Float64, 2*L)
                x = [Int(i<p) for i in B]
            else
                x = copy(probArray[t, :])
            end
            randVec = zeros(1, 4*L+3)    
            largeU = zeros(4*L+2, 4*L+3)        
            modTwo = t%2        
            if modTwo==1
                GStabInit = copy(GStab)                        
                for i in 1:L
                    if givenU
                        randInt = Int(randIntVec1[(t-1)*L + i])
                    else
                        randInt = Int(rand(0:719))
                    end
                    tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                    largeU[4*(i-1)+1:4*(i-1)+4, 4*(i-1)+1:4*(i-1)+4] = copy(tempU)  
                
                    largeU[4*(i-1)+1:4*(i-1)+4, 4*L+3] = copy(randBit[1:4, 1])
                    GStab[:, 4*(i-1)+1:4*(i-1)+4] = GStab[:, 4*(i-1)+1:4*(i-1)+4]*tempU.%2
                end  
                largeU[4*L+1, 4*L+1] = 1            
                largeU[4*L+2, 4*L+2] = 1
                GStabLargeU = (GStabInit[:, 1:4*L+2]*largeU).%2
            
                GStab = PauliTransform(GStabInit, largeU)
            else
                GStabInit = copy(GStab)
                for i in 1:Int(L)-1
                    if givenU
                        randInt = Int(randIntVec1[(t-1)*L + i])
                    else
                        randInt = Int(rand(0:719))
                    end
                    tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                    largeU[4*(i-1)+3:4*(i-1)+6, 4*(i-1)+3:4*(i-1)+6] = copy(tempU)
                    largeU[4*(i-1)+3:4*(i-1)+6, 4*L+3] = copy(randBit[1:4, 1])
                end
                if givenU
                    randInt = Int(randIntVec1[t*L])
                else
                    randInt = Int(rand(0:719))
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                largeU[1:2, 1:2] = copy(tempU[1:2, 1:2])            
                largeU[1:2, 4*L-1:4*L] = copy(tempU[1:2, 3:4])
                largeU[4*L-1:4*L, 1:2] = copy(tempU[3:4, 1:2])
                largeU[4*L-1:4*L, 4*L-1:4*L] = copy(tempU[3:4, 3:4])
            
                largeU[1:2, 4*L+3] = copy(randBit[1:2, 1])
                largeU[4*L-1:4*L, 4*L+3] = copy(randBit[3:4, 1])
            
                largeU[4*L+1, 4*L+1] = 1                
                largeU[4*L+2, 4*L+2] = 1
            
                GStab = PauliTransform(GStabInit, largeU)
            end
        end
        A = [1, 4*Int(L)]
        EEBefore = Entropy(GStab[1:2*L+1, :], A)
    end            
    GStab1 = zeros(size(GStab))
    GStab2 = zeros(size(GStab))    
    
    purifyTime = 0
    # Time evolution WITH measurement #
    ###################################
    for t in 1:T        
        if !(givenRand)
            B = rand(Float64, 2*L)            
            x = [Int(i<p) for i in B]
        else
            x = copy(probArray[t, :])
        end

        randVec = zeros(1, 4*L+3)
        largeU = zeros(4*L+2, 4*L+3)
        
        for i in 1:2*L
            if Bool(x[i])
                randVec[2*i-1] = 1
                
                measurePauliRes = convQCliffMeasure(GStab, randVec)                      
                GStab = copy(measurePauliRes[1])  
                measureResVec[t, i] = copy(measurePauliRes[2])
                
                randVecPauli = convertPauliRep(randVec)                                        
                randVec[2*i-1] = 0
            end
        end
        # Detecting whether Ancilla is purified or not
        
        if purifyTime==0
            EE = Entropy(GStab[1:2*L+1, :], A)
            measurePauliX = convQCliffMeasure(GStab, ancillaVecX)
            deterX = measurePauliX[3]

            measurePauliY = convQCliffMeasure(GStab, ancillaVecY)
            deterY = measurePauliY[3]
                
            measurePauliZ = convQCliffMeasure(GStab, ancillaVecZ)
            deterZ = measurePauliZ[3]
            deterministic = [deterX, deterY, deterZ]
            if (deterX != 0 || deterY != 0 || deterZ != 0)
                EE = Entropy(GStab[1:2*L+1, :], A)
                purifyTime = t
            end
        end
                        
        GStabInit = copy(GStab)
        
        modTwo = t%2
        if modTwo==1
            for i in 1:L
                if Bool(givenU)
                    randInt = Int(randIntVec2[(t-1)*L + i])
                else
                    randInt = Int(rand(0:719))
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                #randBit = rand(0:1, 4, 1)
                largeU[4*(i-1)+1:4*(i-1)+4, 4*(i-1)+1:4*(i-1)+4] = copy(tempU)  
                largeU[4*(i-1)+1:4*(i-1)+4, 4*L+3] = copy(randBit[1:4, 1])
                GStab[:, 4*(i-1)+1:4*(i-1)+4] = GStab[:, 4*(i-1)+1:4*(i-1)+4]*tempU .% 2
            end   
            largeU[4*L+1, 4*L+1] = 1                
            largeU[4*L+2, 4*L+2] = 1
            GStabLargeU = GStabInit[:, 1:4*L+2]*largeU.%2
            GStab = PauliTransform(GStabInit, largeU)
        else
            GStabInit = copy(GStab)                                    
            for i in 1:Int(L)-1
                if givenU
                    randInt = Int(randIntVec2[(t-1)*L + i])
                else
                    randInt = rand(0:719)
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]                      
                largeU[4*(i-1)+3:4*(i-1)+6, 4*(i-1)+3:4*(i-1)+6] = copy(tempU)
                randbit = [0, 0, 0, 0]                
                largeU[4*(i-1)+3:4*(i-1)+6, 4*L+3] = copy(randBit[1:4, 1])
            end
            if givenU
                randInt = Int(randIntVec2[t*L])
            else
                randInt = rand(0:719)   
            end
            tempU = randU[4*randInt+1:4*randInt+4, 1:4]
            randbit = [0, 0, 0, 0]            
            largeU[1:2, 1:2] = copy(tempU[1:2, 1:2])            
            largeU[1:2, 4*L-1:4*L] = copy(tempU[1:2, 3:4])
            largeU[4*L-1:4*L, 1:2] = copy(tempU[3:4, 1:2])
            largeU[4*L-1:4*L, 4*L-1:4*L] = copy(tempU[3:4, 3:4])
            
            largeU[1:2, 4*L+3] = copy(randBit[1:2, 1])
            largeU[4*L-1:4*L, 4*L+3] = copy(randBit[3:4, 1])
            
            largeU[4*L+1, 4*L+1] = 1                
            largeU[4*L+2, 4*L+2] = 1
            
            GStab = PauliTransform(GStabInit, largeU)

        end
    end
    
    A = [1, 4*Int(L)]
    EE = Entropy(GStab[1:2*L+1, :], A)

    finalGStabX = copy(GStab)
    finalGStabY = copy(GStab)
    finalGStabZ = copy(GStab)
    convFinalGStabX  = convertStabRep(finalGStabX)
    convFinalGStabY  = convertStabRep(finalGStabY)          
    convFinalGStabZ  = convertStabRep(finalGStabZ)          
    
    measurePauliX = QCliffMeasure(convFinalGStabX, ancilVecXPauli) 
    deterX = measurePauliX[3]        
    finalGStabX = measurePauliX[1]
    finalMeasureX = measurePauliX[2]
    deterX = measurePauliX[3]

    measurePauliY = QCliffMeasure(convFinalGStabY, ancilVecYPauli)
    finalGStabY = measurePauliY[1]
    finalMeasureY = measurePauliY[2]
    deterY = measurePauliY[3]

    measurePauliZ = QCliffMeasure(convFinalGStabZ, ancilVecZPauli)    
    finalGStabZ = measurePauliZ[1]
    finalMeasureZ = measurePauliZ[2]
    deterZ = measurePauliZ[3]
    
    deterministic = [deterX, deterY, deterZ]
    finalMeasures = [finalMeasureX, finalMeasureY, finalMeasureZ]
    
    if purifyTime == 0
        purifyTime = NaN
    end
    return GStab, measureResVec, finalMeasures, deterministic, purifyTime, EE     
end

function randArrGeneration(L, p, T, Ncircuit, Nbatch, Scramble)     
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end
    println("ScrambleLabel = ", ScrambleLabel)
    println("Nb = ", Nbatch)
    if Nbatch==false
        Nbatch = 1
    end
    for n in 1:Nbatch
        print("n = ", n)
        randVecArr1 = zeros(Ncircuit, L*T);
        randVecArr2 = zeros(Ncircuit, L*T);
        probArr = zeros(T, 2*L);
        probArrMat = zeros(Ncircuit, T, 2*L);
    
        for c in 1:Ncircuit
            if c%100 == 0
                println("c = ", c)
            end
            randVec1 = rand(0:719, 1, L*T)
            randVec2 = rand(0:719, 1, L*T)
            randVecArr1[c, :] = copy(randVec1[1, 1:L*T])
            randVecArr2[c, :] = copy(randVec2[1, 1:L*T]) 
            for t in 1:T
                B = rand(Float64, 2*L)    
                for i in 1:2*L
                    probArr[t, i] = Int(B[i]<p)
                end
            end
            probArrMat[c, :, :] = copy(probArr)
        end
        rshpProb = reshape(probArrMat, (Ncircuit*T, 2*L))
        if Nbatch>1  
            print("Nb>1")            
            try
                save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", probArrMat)
                save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr1)
                save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr2)
            catch y                
                if isa(y, ArgumentError)
                    save("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", probArrMat)
                    save("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr1)
                    save("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr2)
                end
            end
        elseif (Nbatch ==1) || (Nbatch == false)
            print("elseif")
            try            
                save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", probArrMat)
                save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr1)
                save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr2)            
            catch y
                if isa(y, ArgumentError)
                    save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", probArrMat)
                    save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr1)
                    save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr2)            
                end
            end
        end
    end    
end    






function QCliffTimeEvolveAncilla(L, p, T, withScramble, randIntVec1 = [], randIntVec2 = [], probArray = [])
    # We have 2*L+1 qubits with the same number of Stabilizers. Of these number of qubits, one is an ancilla qubit which is located in the middle of the system's qubits. 
    # Time evolution is only applied to all the qubits except the ancilla qubit. 
    # 
    # print('probArray \n', probArray)
    # size(Ginit) = (4*L+2)*(4L+2)
    # The ancilla qubit is put at the end of the string of the qubits. 
    
    # We first time evolve a Clifford circuit without performing any measurements. Next, we make a measurement 
    # on the middle qubit and next, we maximally entangle the middle qubit of the stabilizer state with a 
    # reference qubit by forming a Bell pair. 
    
    T1 = 4*L # Time evolved to create a randomly entangled volume state. 
    T2 = T # Time to evolve the circuit after entangling the middle qubit with the reference qubit. 
    Ginit = zeros(2*L+1, 4*L+3)
    ancillaQbit = 4*L
    middleQbit = 2*L
    ancilla = 0
    initStabDestab = initProdState(2*L+1, ancilla) # Initial state is a product state along the X axis
    
    Ginit = initStabDestab[1:2*L+1, :] # Initial state is a product state along the X axis    
    
    Ginit[L, ancillaQbit+2] = 0   # X=0
    Ginit[L, ancillaQbit+1] = 1   # Z=1    
    
    Ginit[2*L+1, ancillaQbit+2] = 1 # X=1
    Ginit[2*L+1, ancillaQbit+1] = 0 # Z=0
    Ginit[2*L+1, middleQbit-1] = 0  # Z=0
    Ginit[2*L+1, middleQbit] = 1  # X=1

    GStab = zeros(size(Ginit))
    GStab = copy(Ginit)

    if size(probArray)[1] == 0
        givenRand = false
    else
        givenRand = true
    end

    if size(randIntVec1)[1] == 0
        givenU = false
    else
        givenU = true
    end    
    
    GStabLastGate = zeros(2*L+1, 4)
    measureResVec = zeros(T, 2*L+1)
    measureResVec[:, :] .= NaN
    
    ancillMeasure = zeros(1, 4*L+3)
    ancillMeasure[1, 4*L+1] = 1      # Measuring vector index
    ancilMeasPauli = convertPauliRep(ancillMeasure)
    ancillaVecX = zeros(1, 4*L+3)
    ancillaVecX[4*L+2] = 1 
    ancilVecXPauli = convertPauliRep(ancillaVecX)
    
    ancillaVecY = zeros(1, 4*L+3)
    ancillaVecY[4*L+2] = 1 
    ancillaVecY[4*L+1] = 1 
    ancilVecYPauli = convertPauliRep(ancillaVecY)    
    
    ancillaVecZ = zeros(1, 4*L+3)
    ancillaVecZ[4*L+1] = 1  
    ancilVecZPauli = convertPauliRep(ancillaVecZ)    
    
    # Time evolution WITHOUT measurement #
    ######################################

    # We start by applying unitary time evolution without any measurements. 
    
    pop = [0, 1]
    weights = [1-p, p]
    randBit = [0, 0, 0, 0]    
    A = [1, 4*Int(L)]
    randU = RandUnitaries()
    TScramble = 4*L; 
    if withScramble

        for t in 1:TScramble
            if !(givenRand)
                B = rand(Float64, 2*L)
                x = [Int(i<p) for i in B]
            else
                x = copy(probArray[t, :])
            end
            randVec = zeros(1, 4*L+3)    
            largeU = zeros(4*L+2, 4*L+3)        
            modTwo = t%2        
            if modTwo==1
                GStabInit = copy(GStab)                        
                for i in 1:L
                    if givenU
                        randInt = randIntVec1[(t-1)*L + i]
                    else
                        randInt = rand(0:719)
                    end
                    tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                    largeU[4*(i-1)+1:4*(i-1)+4, 4*(i-1)+1:4*(i-1)+4] = copy(tempU)                  
                    largeU[4*(i-1)+1:4*(i-1)+4, 4*L+3] = copy(randBit[1:4, 1])
                    GStab[:, 4*(i-1)+1:4*(i-1)+4] = GStab[:, 4*(i-1)+1:4*(i-1)+4]*tempU.%2
                end  
                largeU[4*L+1, 4*L+1] = 1            
                largeU[4*L+2, 4*L+2] = 1
                GStabLargeU = (GStabInit[:, 1:4*L+2]*largeU).%2            
                GStab = PauliTransform(GStabInit, largeU)
            else    
                GStabInit = copy(GStab)
                for i in 1:Int(L)-1
                    if givenU
                        randInt = randIntVec1[(t-1)*L + i]
                    else
                        randInt = rand(0:719)
                    end
                    tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                    largeU[4*(i-1)+3:4*(i-1)+6, 4*(i-1)+3:4*(i-1)+6] = copy(tempU)
                    largeU[4*(i-1)+3:4*(i-1)+6, 4*L+3] = copy(randBit[1:4, 1])
                end
                if givenU
                    randInt = randIntVec1[t*L]
                else
                    randInt = rand(0:719)
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                largeU[1:2, 1:2] = copy(tempU[1:2, 1:2])            
                largeU[1:2, 4*L-1:4*L] = copy(tempU[1:2, 3:4])
                largeU[4*L-1:4*L, 1:2] = copy(tempU[3:4, 1:2])
                largeU[4*L-1:4*L, 4*L-1:4*L] = copy(tempU[3:4, 3:4])
            
                largeU[1:2, 4*L+3] = copy(randBit[1:2, 1])
                largeU[4*L-1:4*L, 4*L+3] = copy(randBit[3:4, 1])
            
                largeU[4*L+1, 4*L+1] = 1                
                largeU[4*L+2, 4*L+2] = 1            
                GStab = PauliTransform(GStabInit, largeU)
            end
        end
        A = [1, 4*Int(L)]
        EEBefore = Entropy(GStab[1:2*L+1, :], A)
    end    
        
    GStab1 = zeros(size(GStab))
    GStab2 = zeros(size(GStab))    
    
    purifyTime = 0
    # Time evolution WITH measurement #
    ###################################

    for t in 1:T
        if !(givenRand)
            B = rand(Float64, 2*L)            
            x = [Int(i<p) for i in B]
        else
            x = copy(probArray[t, :])
        end        
        randVec = zeros(1, 4*L+3)
        largeU = zeros(4*L+2, 4*L+3)
                
        for i in 1:2*L 
            if Bool(x[i])
                randVec[2*i-1] = 1
                
                measurePauliRes = convQCliffMeasure(GStab, randVec)                      
                GStab = copy(measurePauliRes[1])                    
                measureResVec[t, i] = copy(measurePauliRes[2])
                    
                randVecPauli = convertPauliRep(randVec)                                        
                randVec[2*i-1] = 0
            end
        end
        # Detecting whether Ancilla is purified or not
        
        if purifyTime==0
            EE = Entropy(GStab[1:2*L+1, :], A)

            tempGStabX = copy(GStab)
            tempGStabY = copy(GStab)
            tempGStabZ = copy(GStab)
            
            measurePauliX = convQCliffMeasure(tempGStabX, ancillaVecX)
            deterX = measurePauliX[3]

            measurePauliY = convQCliffMeasure(tempGStabY, ancillaVecY)
            deterY = measurePauliY[3]
                
            measurePauliZ = convQCliffMeasure(tempGStabZ, ancillaVecZ)
            deterZ = measurePauliZ[3]
            deterministic = [deterX, deterY, deterZ]
            if (deterX != 0 || deterY != 0 || deterZ != 0)
                EE = Entropy(GStab[1:2*L+1, :], A)
                purifyTime = t
            end
        end
                        
        GStabInit = copy(GStab)
        
        modTwo = t%2
        if modTwo==1
            for i in 1:L
                if Bool(givenU)
                    randInt = Int(randIntVec2[(t-1)*L + i])
                else
                    randInt = rand(0:719)
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]
                #randBit = rand(0:1, 4, 1)
                largeU[4*(i-1)+1:4*(i-1)+4, 4*(i-1)+1:4*(i-1)+4] = copy(tempU)  
                largeU[4*(i-1)+1:4*(i-1)+4, 4*L+3] = copy(randBit[1:4, 1])
                GStab[:, 4*(i-1)+1:4*(i-1)+4] = GStab[:, 4*(i-1)+1:4*(i-1)+4]*tempU .% 2
            end   
            largeU[4*L+1, 4*L+1] = 1                
            largeU[4*L+2, 4*L+2] = 1
            GStabLargeU = GStabInit[:, 1:4*L+2]*largeU.%2
            GStab = PauliTransform(GStabInit, largeU)
        else
            GStabInit = copy(GStab)                                    
            for i in 1:Int(L)-1
                if givenU
                    randInt = Int(randIntVec2[(t-1)*L + i])
                else
                    randInt = rand(0:719)
                end
                tempU = randU[4*randInt+1:4*randInt+4, 1:4]                      
                largeU[4*(i-1)+3:4*(i-1)+6, 4*(i-1)+3:4*(i-1)+6] = copy(tempU)
                randbit = [0, 0, 0, 0]                
                largeU[4*(i-1)+3:4*(i-1)+6, 4*L+3] = copy(randBit[1:4, 1])
            end
            if givenU
                randInt = Int(randIntVec2[t*L])
            else
                randInt = rand(0:719)   
            end
            tempU = randU[4*randInt+1:4*randInt+4, 1:4]
            randbit = [0, 0, 0, 0]            
            
            largeU[1:2, 1:2] = copy(tempU[1:2, 1:2])            
            largeU[1:2, 4*L-1:4*L] = copy(tempU[1:2, 3:4])
            largeU[4*L-1:4*L, 1:2] = copy(tempU[3:4, 1:2])
            largeU[4*L-1:4*L, 4*L-1:4*L] = copy(tempU[3:4, 3:4])
            
            largeU[1:2, 4*L+3] = copy(randBit[1:2, 1])
            largeU[4*L-1:4*L, 4*L+3] = copy(randBit[3:4, 1])
            
            largeU[4*L+1, 4*L+1] = 1                
            largeU[4*L+2, 4*L+2] = 1
            
            GStab = PauliTransform(GStabInit, largeU)
        end
    end
    
    A = [1, 4*Int(L)]
    EE = Entropy(GStab[1:2*L+1, :], A)
    finalGStabX = copy(GStab)
    finalGStabY = copy(GStab)
    finalGStabZ = copy(GStab)
    convFinalGStabX  = convertStabRep(finalGStabX)
    convFinalGStabY  = convertStabRep(finalGStabX)          
    convFinalGStabZ  = convertStabRep(finalGStabX)          
                
    measurePauliX = QCliffMeasure(convFinalGStabX, ancilVecXPauli) 
    deterX = measurePauliX[3]    
    
    finalGStabX = measurePauliX[1]
    finalMeasureX = measurePauliX[2]
    deterX = measurePauliX[3]

    measurePauliY = QCliffMeasure(convFinalGStabY, ancilVecYPauli)
    finalGStabY = measurePauliY[1]
    finalMeasureY = measurePauliY[2]
    deterY = measurePauliY[3]

    measurePauliZ = QCliffMeasure(convFinalGStabZ, ancilVecZPauli)    
    finalGStabZ = measurePauliZ[1]
    finalMeasureZ = measurePauliZ[2]
    deterZ = measurePauliZ[3]
    
    deterministic = [deterX, deterY, deterZ]
    finalMeasures = [finalMeasureX, finalMeasureY, finalMeasureZ]
    
    for i in 1:size(GStab)[1]
        #println("GStab[$i] = ", GStab[i, :])
    end
    if purifyTime == 0
        #purifyTime = NaN
    end
    return GStab, measureResVec, finalMeasures, deterministic, purifyTime, EE 
end

function randArrGenerationARGS(args)
    # Generates a Random Clifford Circuit 
    L = parse(Int, args[1])
    p = parse(Float64, args[2])
    T = parse(Int, args[3])
    Ncircuit = parse(Int, args[4])
    println("args[5] = ", typeof(args[5]))
    Nbatch = parse(Int, args[5])    
    
    println("Nbatch = ", Nbatch)
    
    nb = parse(Int, args[6])
    cVec = false
    try
        cVec = parse(Int, args[7])
        catch y.
        if isa(y, ArgumentError)
            cVec = false
        end
    end
    
    Ncirc1 = parse(Int, args[8])
    Ncirc2 = parse(Int, args[9])    
    
    Scramble = parse(Int, args[10])
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 
    
    probArrMat = zeros(Ncircuit, T, 2*L);
    rv1 = zeros(Ncircuit, L*T);
    rv2 = zeros(Ncircuit, L*T);
    
    for n in 1:Nbatch
        randVecArr1 = zeros(Ncircuit, L*T);
        randVecArr2 = zeros(Ncircuit, L*T);
        probArr = zeros(T, 2*L);
        probArrMat = zeros(Ncircuit, T, 2*L);
    
        for c in 1:Ncircuit
            if c%100 == 0
                println("c = ", c)
            end
            randVec1 = rand(0:719, 1, L*T)
            randVec2 = rand(0:719, 1, L*T)
            randVecArr1[c, :] = copy(randVec1[1, 1:L*T])
            randVecArr2[c, :] = copy(randVec2[1, 1:L*T])    
            for t in 1:T
                B = rand(Float64, 2*L)    
                for i in 1:2*L
                    probArr[t, i] = Int(B[i]<p)
                end
            end
            probArrMat[c, :, :] = copy(probArr)
        end
        rshpProb = reshape(probArrMat, (Ncircuit*T, 2*L))
        if Nbatch>1  
            try
                save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", probArrMat)
                save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr1)
                save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr2)
            catch y                
                if isa(y, ArgumentError)
                    save("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", probArrMat)
                    save("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr1)
                    save("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld", "data", randVecArr2)
                end
            end
        elseif (Nbatch ==1) || (Nbatch == false)
            try            
                save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", probArrMat)
                save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr1)
                save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr2)            
                CSV.write("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(randVecArr1)), writeheader=true)            
                CSV.write("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(randVecArr2)), writeheader=true)                        
            catch y
                if isa(y, ArgumentError)
                    save("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", probArrMat)
                    save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr1)
                    save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", randVecArr2)            
                    CSV.write("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(randVecArr1)), writeheader=true)            
                    CSV.write("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(randVecArr2)), writeheader=true)                        
                end
            end
        end
    end
end    




function determinePurifyTime(L, p, T, Scramble, Ncircuit, Nbatch=false, nb=false, cVec = false, csv=false)
#    determinePurifyTime
    
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end
    
    if csv
        rv1 = zeros(Ncircuit, L*T);""
        rv2 = zeros(Ncircuit, L*T);

        ## Loading randomVecs and ProbArray of measurements:
    
        csv_reader = CSV.File("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")

        df = DataFrame(csv_reader)
        println(size(df, 1),',', size(df, 2))
    
        rv1 = zeros(size(df, 1), size(df, 2));

        println("size rv1 = ", size(rv1), size(rv2))

        for i in 1:size(rv1)[1]
            #println("i = ", i)
            for j in 1:size(rv1)[2]
                rv1[i, j] = df[i, j]
            end
        end
        println("rv1[1, 1:10] = ", rv1[1, 1:10])
        csv_reader = CSV.File("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")

        df = DataFrame(csv_reader)
        println(size(df, 1),',', size(df, 2))
    
        rv2 = zeros(size(df, 1), size(df, 2));
        for i in 1:size(rv2)[1] 
            #println("i = ", i)        
            for j in 1:size(rv2)[2]
                rv2[i, j] = df[i, j]
            end
        end
        save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", rv1)
        save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", rv2)                
    end

    probArrMat = zeros(Ncircuit, T, 2*L);
    rv1 = zeros(Ncircuit, L*T);
    rv2 = zeros(Ncircuit, L*T);
    Ncirc1 = 1
    Ncirc2 = Ncircuit
    purifyTimeVec = zeros(Ncircuit)

    #Scramble = parse(Int, args[10])    
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 
    keepTraj = true
    if Nbatch==false || Nbatch == 1
        try                
            probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
            rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
            rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
        catch y                
            if isa(y, ArgumentError)
                probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")["data"]
                rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")["data"]
                rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")["data"]                
            end
        end
            
    else
        println("nb = ", nb)
        probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
        rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
        rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]       
    end
    
    println("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")    
    save("probArrMatDeterminePTL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld", "data", probArrMat)
    println("loop")

    if cVec==false
        Threads.@threads for c in Ncirc1:Ncirc2
            if c%10==0
                println("c = ", c)
            end
            #A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], rv2[c, :], probArrMat[c, :, :]);
            if useQCliffOrQClifford==1
                A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :]);
            else
                A = QCliffordAncilla(L, p, T, Bool(Scramble), keepTraj, rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :])            
            end
            
            #A = QCliffordAncilla(L, p, T, Bool(Scramble), keepTraj, rv1[c, :], 
            #    rv2[c, :], probArrMat[c, :, :])            
            
            flush(stdout)
            println("c = ", c, " pure time = ", A[5])

            purifyTimeVec[c] = A[5]
            #print(purifyTimeVec[c])
        end            
    else 
        Threads.@threads for c in cVec
            if c%10==0
                println("c = ", c)
            end
            #A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], rv2[c, :], probArrMat[c, :, :]);
            #A = QCliffordAncilla(L, p, T, Bool(Scramble), keepTraj, rv1[c, :], 
            #    rv2[c, :], probArrMat[c, :, :]);   
            if useQCliffOrQClifford==1
                A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :]);
            else
                A = QCliffordAncilla(L, p, T, Bool(Scramble), keepTraj, rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :])            
            end
            

            flush(stdout)
            println("c = ", c, " pure time = ", A[5])
            println("final measure = ", A[3])
            println("determ = ", A[4])            
            purifyTimeVec[c] = A[5]
            #print(purifyTimeVec[c])
        end
    end
    print(transpose(purifyTimeVec))
    #    determinePurifyTime
    #A = DataFrame(transpose(purifyTimeVec))
    if cVec==false 
        if Nbatch==false || Nbatch == 1
            if Ncirc1==1
                println("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")
                save("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", purifyTimeVec)
                #CSV.write("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                 
            elseif (Ncirc1!=1 || Ncirc2!=Ncircuit)
                println("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-T$T$ScrambleLabel.jld")
                #println("Ncirc1!=1 || Ncirc2!=Ncircuit")
                save("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-T$T$ScrambleLabel.jld", "data", purifyTimeVec)
                #CSV.write("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                                 
            end
        elseif Nbatch!=false 
            if Ncirc1==1 && Ncirc2==Ncircuit
                println("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")
                save("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld", "data", purifyTimeVec)            
                #CSV.write("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                                                 
            elseif (Ncirc1!=1 || Ncirc2!=Ncircuit)
                println("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")
                save("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld", "data", purifyTimeVec)
                #CSV.write("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                                                                 
            end 
        end
       # println("purifyTimeVec", purifyTimeVec)
    else 
        save("purifyTimeL$L-Ncirc$Ncircuit-p$p-cVec$cVec$ScrambleLabel.jld", "data", purifyTimeVec)                
        CSV.write("purifyTimeL$L-Ncirc$Ncircuit-p$p-cVec$cVec$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                                                                         
    end

    return purifyTimeVec 
end




function determinePurifyTimeARGS(args)
    
    L = parse(Int, args[1])
    p = parse(Float64, args[2])
    T = parse(Int, args[3])
    Ncircuit = parse(Int, args[4])
    println("args[5] = ", typeof(args[5]))
    Nbatch = false
    try
        println("parse") 
        println("parse(Int, args[5]) = ", parse(Int, args[5]))
        Nbatch = parse(Int, args[5])
    catch y
        if isa(y, ArgumentError)
            Nbatch = false
            println("x is a string")
        end
    end    
    println("Nbatch = ", Nbatch)
    
    nb = parse(Int, args[6])
    
    #severalJob = false
    #severalJob = parse(Bool, args[7])
    #println("severalJOB = ", severalJob)
    #if severalJob ==true
    #    job_id=ENV["SLURM_ARRAY_TASK_ID"]
    #end
    cVec = false
    try
        cVec = parse(Int, args[7])
    catch y
        if isa(y, ArgumentError)
            cVec = false
        end
    end
    
    Ncirc1 = parse(Int, args[8])
    Ncirc2 = parse(Int, args[9])    
    
    Scramble = parse(Int, args[10])    
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 
    
    probArrMat = zeros(Ncircuit, T, 2*L);
    rv1 = zeros(Ncircuit, L*T);
    rv2 = zeros(Ncircuit, L*T);
    #Ncirc1 = 1
    #Ncirc2 = Ncircuit
    purifyTimeVec = zeros(Ncircuit)

    if Nbatch==false || Nbatch == 1
        probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
        rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
        rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
    else
        println("nb = ", nb)
        probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
        rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
        rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
        println("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")
        save("probArrMatDeterminePTL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld", "data", probArrMat)        
    end
    
    println(size(rv1))
    println(size(rv2))
    println(size(probArrMat))
    
    println("loop")
    println("cVec = ", cVec)
    
    if cVec==false
        Threads.@threads for c in Ncirc1:Ncirc2
            if c%10==0
                println("c = ", c)
            end
            if useQCliffOrQClifford==1
                A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :]);
            else
                A = QCliffordAncilla(L, p, T, Bool(Scramble), keepTraj, rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :])            
            end
            
            flush(stdout)
            println("c = ", c, " pure time = ", A[5])

            purifyTimeVec[c] = A[5]

        end            
    else 
        Threads.@threads for c in cVec
            if c%10==0
                println("c = ", c)
            end
            #A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], rv2[c, :], probArrMat[c, :, :]);
            if useQCliffOrQClifford==1
                A = QCliffTimeEvolveAncilla(L, p, T, Bool(Scramble), rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :]);
            else
                A = QCliffordAncilla(L, p, T, Bool(Scramble), keepTraj, rv1[c, :], 
                    rv2[c, :], probArrMat[c, :, :])            
            end
            
            flush(stdout)
            println("c = ", c, " pure time = ", A[5])
            println("final measure = ", A[3])
            println("determ = ", A[4])            
            purifyTimeVec[c] = A[5]
            #print(purifyTimeVec[c])
        end
    end
    if cVec==false 
        if Nbatch==false || Nbatch == 1
            if Ncirc1==1
                println("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")
                save("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", purifyTimeVec)
                CSV.write("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                 
            elseif (Ncirc1!=1 || Ncirc2!=Ncircuit)
                println("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-T$T$ScrambleLabel.jld")
                #println("Ncirc1!=1 || Ncirc2!=Ncircuit")
                save("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-T$T$ScrambleLabel.jld", "data", purifyTimeVec)
                CSV.write("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)
            end
        elseif Nbatch!=false 
            if Ncirc1==1 && Ncirc2==Ncircuit
                println("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")
                save("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld", "data", purifyTimeVec)            
                CSV.write("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                                                 
            elseif (Ncirc1!=1 || Ncirc2!=Ncircuit)
                println("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")
                save("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld", "data", purifyTimeVec)
                CSV.write("purifyTimeL$L-Ncirc$Ncircuit-Ncirc1st$Ncirc1-NcircLast$Ncirc2-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                                                                 
            end 
        end
       # println("purifyTimeVec", purifyTimeVec)
    else 
        save("purifyTimeL$L-Ncirc$Ncircuit-p$p-cVec$cVec$ScrambleLabel.jld", "data", purifyTimeVec)                
        CSV.write("purifyTimeL$L-Ncirc$Ncircuit-p$p-cVec$cVec$ScrambleLabel.csv",  DataFrame(transpose(purifyTimeVec)), writeheader=true)                                                                         
    end

    return purifyTimeVec 
end 


function circIndArrGeneration(L, p, T, Scramble, Ncircuit, Nbatch, NpureT, NcircPT)    
    
    #Scramble = parse(Int, args[10])    
    println("Scramble = ", Scramble)
    ScrambleLabel = ""
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 
     
    
    if Nbatch==false
        purifyTimeVec = zeros(Ncircuit)
        #purifyTimeVec = load("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
        purifyTimeVec = load("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
    elseif Nbatch>1
        purifyTimeVec = zeros(Ncircuit*Nbatch)
        for n = 1:Nbatch
            purifyTime = load("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld")["data"]
            println("purifyTime = ", purifyTime)
            if n==1
                purifyTimeVec = vcat(purifyTime)
            elseif n>1
                purifyTimeVec = vcat(purifyTimeVec, purifyTime)
            end
        end
    end
    
    println("purifyTimeVec = ", purifyTimeVec)
    #NpureT = 10; # we take the first 10 purification times for learning.
    #NcircPT = 20; # we consider 10 circuit realization for each purification time.
    circIndArr = zeros(Int64, NcircPT, NpureT);
    lastFoundInd = zeros(NpureT)
    for i in 1:size(purifyTimeVec)[1]

        
#0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 
#3.0, 1.0, 1.0, 3.0, 1.0, 2.0, 2.0, 3.0, 5.0, 8.0, 1.0, 1.0, 
#3.0, 17.0, 5.0, 1.0, 2.0, 2.0, 3.0        
        
        if purifyTimeVec[i] <= NpureT && purifyTimeVec[i] > 0
            #println(purifyTimeVec[i], ' ', i)
            lastFoundInd[Int(purifyTimeVec[i])] += 1
            if lastFoundInd[Int(purifyTimeVec[i])] <= NcircPT
                #println("inside if")
                circIndArr[Int(lastFoundInd[Int(purifyTimeVec[i])]), Int(purifyTimeVec[i])] = Int(i)
                println("i, purifyTimeVec[i],  circIndArr[Int(lastFoundInd[Int(purifyTimeVec[i])]), 
                    Int(purifyTimeVec[i])] = ", i, ' ', purifyTimeVec[i], ' ', 
                    circIndArr[Int(lastFoundInd[Int(purifyTimeVec[i])]), Int(purifyTimeVec[i])])
            end
        else
            continue
        end
    end

    println("circIndArr[:, PT] = ", circIndArr[:, :])
    if Nbatch==false
        save("circIndArrL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", circIndArr)
        CSV.write("circIndArrL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(circIndArr)), writeheader=true)
    elseif Nbatch>1
        save("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.jld", "data", circIndArr)
        CSV.write("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.csv",  DataFrame(transpose(circIndArr)), writeheader=true)        
    end
    return circIndArr
end

function circIndArrGenerationARGS(args)
    
    L = parse(Int, args[1])
    p = parse(Float64, args[2])
    T = parse(Int, args[3])
    Ncircuit = parse(Int, args[4])
    println("args[5] = ", typeof(args[5]))
    Nbatch = false
    try
        println("parse") 
        println("parse(Int, args[5]) = ", parse(Int, args[5]))
        Nbatch = parse(Int, args[5])
    catch y
        if isa(y, ArgumentError)
            Nbatch = false
            println("x is a string")
        end
    end    
    println("Nbatch = ", Nbatch)
    
    
    NpureT = parse(Int, args[6]);
    NcircPT = parse(Int, args[7]);
    
    Scramble = parse(Int, args[8])    
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 
    
    if Nbatch==false
        purifyTimeVec = zeros(Ncircuit)
        #purifyTimeVec = load("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
        purifyTimeVec = load("purifyTimeL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
    elseif Nbatch>1
        purifyTimeVec = zeros(Ncircuit*Nbatch)
        for n = 1:Nbatch
            purifyTime = load("purifyTimeL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$n$ScrambleLabel.jld")["data"]
            println("purifyTime = ", purifyTime)
            if n==1
                purifyTimeVec = vcat(purifyTime)
            elseif n>1
                purifyTimeVec = vcat(purifyTimeVec, purifyTime)
            end
        end
    end
    
    println("purifyTimeVec = ", purifyTimeVec)
    circIndArr = zeros(Int64, NcircPT, NpureT);
    lastFoundInd = zeros(NpureT)
    for i in 1:size(purifyTimeVec)[1]
        #println(purifyTimeVec[i])
        if purifyTimeVec[i] <= NpureT && purifyTimeVec[i] > 0
            #println(purifyTimeVec[i], ' ', i)
            lastFoundInd[Int(purifyTimeVec[i])] += 1
            if lastFoundInd[Int(purifyTimeVec[i])] <= NcircPT
                #println("inside if")
                circIndArr[Int(lastFoundInd[Int(purifyTimeVec[i])]), Int(purifyTimeVec[i])] = Int(i)
            end
        else
            continue
        end
    end

    println("circIndArr[:, PT] = ", circIndArr[:, :])
    if Nbatch==false
        save("circIndArrL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", circIndArr)
        CSV.write("circIndArrL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv",  DataFrame(transpose(circIndArr)), writeheader=true)                
    elseif Nbatch>1
        save("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.jld", "data", circIndArr)
        CSV.write("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.csv",  DataFrame(transpose(circIndArr)), writeheader=true)                
    end
    return circIndArr
    
end


function randu2(fixedSign=false, t=[], LHalf=[], i=[], randIntVec=[])
    givenU = false
    #println("givenU1 = ", givenU)
    if size(randIntVec)[1] == 0
        givenU = false
    else
        givenU = true
    end    
    
    #println("givenU2 = ", givenU)
    
    if givenU

        randInt = 1+Int(randIntVec[(t-1)*LHalf + i])
        #println("randInt = ", randInt)        
    else
        randInt = 1+rand(0:719)
    end
    if givenU!=false
        u = ru2[randInt]
    elseif givenU==false
        u=rand(ru2)
    end
    
    if fixedSign
        return CliffordOperator(Stabilizer([returnPaul(u[:,i],0x0) for i in 1:size(u,2)]))
    else     
        return CliffordOperator(Stabilizer([returnPaul(u[:,i],rand([0x0,0x2])) for i in 1:size(u,2)]))
    end
end

function QCliffordAncilla(L, p, T, withScramble, keepTraj=false, randIntVec1 = [], randIntVec2 = [], probArray = [])
    
    # MiddleInd is different than the choice considered in QCliffTimeEvolveAncilla. 
    # Here MidInd=2*L, while in QCliffTimeEvolveAncilla is MidInd=L. 
    #println("Int(L/2)")
    givenU1 = false
    if size(randIntVec1)[1] == 0
        givenU1 = false
    else
        givenU1 = true
    end    

    givenU2 = false
    #println("givenU1 = ", givenU)
    if size(randIntVec2)[1] == 0
        givenU2 = false
    else
        givenU2 = true
    end    

    givenMeasure = false
    #println("givenU1 = ", givenU)
    if size(probArray)[1] == 0
        givenMeasure = false
    else
        givenMeasure = true
    end

    
    if !Bool(withScramble)
        ScrambleLabel = ""
    elseif withScramble
        ScrambleLabel = "Scrambled"
    end 
        
    depth=T #4*L
    #println("L:$L, p:$p, T:$depth, ")

    ancillaVecX = zeros(Int8, 1, 4*L+3)
    ancillaVecX[4*L+2] = 1 
    ancilVecXPauli = convertPauliRep(ancillaVecX)
    #println("ancillaVecX = ", ancillaVecX)
    
    ancillaVecY = zeros(Int8, 1, 4*L+3)
    ancillaVecY[4*L+2] = 1 
    ancillaVecY[4*L+1] = 1 
    ancilVecYPauli = convertPauliRep(ancillaVecY)    
    
    ancillaVecZ = zeros(Int8, 1, 4*L+3)
    ancillaVecZ[4*L+1] = 1   
    ancilVecZPauli = convertPauliRep(ancillaVecZ)    
    deterministic = [0, 0, 0]
    deterministicQ = [0, 0, 0]
    measureResVec = zeros(T, 2*L+1)
    measureResVec[:, :] .= NaN
    A = [1, 2*Int(2*L)]
    purifyTime = 0

    purifyX = 0
    finalEE = 0
    fixedSign=false    
        GStab = zeros(2*L+1, 2*(2*L+1)+1)
        state=one(Stabilizer, 2*L+1)
        #middleInd = 2*L
        middleInd = L+1 # New version: 08/02/2022
        refInd = 2*L+1
        if withScramble
            for t=1:depth
                for j=1+t%2:2:2*L
                    if Bool(givenU1)                        
                        tempRandU = randu2(fixedSign, t, Int(L), j ÷ 2, randIntVec1[:])
                    else
                        tempRandU = randu2(fixedSign)
                    end
                    apply!(state, tempRandU, [j,j%(2*L)+1])
                end
            end
            proj = project!(state, single_z(2*L+1, middleInd),keep_result=true,phases=true)
            if proj[3]==2
                apply!(state, single_x(2*L+1,middleInd))
            end
            apply!(state, Hadamard, [refInd])
            apply!(state, CNOT, [refInd, middleInd])
        
        else
            apply!(state, Hadamard, [refInd])
            apply!(state, CNOT, [refInd, middleInd])
        end
        GStab = convBackStab(state)        
        middleEE = Entropy(GStab[1:2*L+1, :], A)
        randArr = zeros(2*L, depth)                
        for t=1:depth
            #println("t = ", t)
            for j=1+t%2:2:2*L
                if Bool(givenU2)
                    tempRandU = randu2(fixedSign, t, Int(L), j ÷ 2, randIntVec2[:])
                else
                    tempRandU = randu2(fixedSign)#randu2(fixedSign)
                end
                apply!(state, tempRandU, [j,j%(2*L)+1])
                GStab = convBackStab(state)
            end            
            if !givenMeasure
                B = rand(Float64, 2*L)            
                x = [Int(i<p) for i in B]
            else
                x = copy(probArray[t, :])
            end   
            for j in 1:2*L 
                if Bool(x[j]) 
                    projection = project!(state, single_z(2*L+1,j),keep_result=keepTraj,phases=true)
                    result = projection[3]
                    if result == 0x0
                        measureRes = 1
                    elseif result == 0x1
                        measureRes = 0.5
                    elseif result == 0x2
                        measureRes = 2
                    elseif result == 0x3
                        measureRes = 1.5  
                    elseif result ==nothing
                        measureRes = rand([0x00, 0x02])
                    end            
                    measureResVec[t, j] = measureRes
                    GStab = convBackStab(state)
                    tempEnt = Entropy(GStab[1:2*L+1, :], A)
                    if tempEnt == 0 && purifyTime == 0 && purifyX == 0
                        purifyTime = t
                        purifyX = j
                    end
                    if tempEnt<0
                        println("break negative entropy = ")
                    end     
                end
            end
            GStab = convBackStab(state)            
            if t==depth
                finalEE = Entropy(GStab[1:2*L+1, :], A)
                if finalEE<0
                    println("break negative entropy = ")
                end
            end            
        end
    finalGStabX = copy(GStab)
    finalGStabY = copy(GStab)
    finalGStabZ = copy(GStab)
    convFinalGStabX  = convertStabRep(finalGStabX)
    convFinalGStabY  = convertStabRep(finalGStabY)          
    convFinalGStabZ  = convertStabRep(finalGStabZ)          
    
    measurePauliX = QCliffMeasure(convFinalGStabX, ancilVecXPauli) 
    deterX = measurePauliX[3]        
    finalGStabX = measurePauliX[1]
    finalMeasureX = measurePauliX[2]
    deterX = measurePauliX[3]
    #println("measure X")
    measurePauliY = QCliffMeasure(convFinalGStabY, ancilVecYPauli)
    finalGStabY = measurePauliY[1]
    finalMeasureY = measurePauliY[2]
    deterY = measurePauliY[3]
    #println("measure Y")
    measurePauliZ = QCliffMeasure(convFinalGStabZ, ancilVecZPauli)    
    finalGStabZ = measurePauliZ[1]
    finalMeasureZ = measurePauliZ[2]
    deterZ = measurePauliZ[3]
    #println("measure Z")    
    deterministic = [deterX, deterY, deterZ]
    finalMeasures = [finalMeasureX, finalMeasureY, finalMeasureZ]
    
    if purifyTime == 0
        purifyTime = NaN
    end
    #println("before return")        
    return GStab, measureResVec, finalMeasures, deterministic, purifyTime, finalEE     
    #return finalEE, purifyTime
end




function createMeasureRecArg(args) #(L, p, T, Nsamp, Ncircuit, Nbatch, NpureT, NcircPT, circIndVec=false)
    #job_id=ENV["SLURM_ARRAY_TASK_ID"]
 
    L = parse(Int, args[1])
    p = parse(Float64, args[2])
    T = parse(Int, args[3])
    PT = parse(Int, args[4])
    Scramble = parse(Int, args[5])    
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 
    
    Nsamp = parse(Int, args[6])
    Ncircuit = parse(Int, args[7])
    println("args[8] = ", typeof(args[8]))
    TEvolve = parse(Int, args[8])
    Nbatch = false
    try
        println("parse") 
        println("parse(Int, args[9]) = ", parse(Int, args[9]))
        Nbatch = parse(Int, args[9])
    catch y
        if isa(y, ArgumentError)
            Nbatch = false
            println("x is a string")
        end
    end
    
    println("Nbatch = ", Nbatch)
    NpureT = parse(Int, args[10])
    NcircPT = parse(Int, args[11])
    circIndVec = parse(Int, args[12])
    
    severalJob = false
    severalJob = parse(Bool, args[13])
    println("severalJOB = ", severalJob)
    if severalJob ==true
        job_id=ENV["SLURM_ARRAY_TASK_ID"]
    end

    csv = parse(Bool, args[14])
    println("csv = ", csv)
    v = false
    try
        v = parse(Int, args[15])
    catch y
        if isa(y, ArgumentError)
            v = false
        end
    end
    
    println(L, p, T, Nsamp, Ncircuit, Nbatch, NpureT, NcircPT, circIndVec, severalJob, csv, v)
    
    if csv
        println("if csv")
        rv1 = zeros(Ncircuit, L*T);
        rv2 = zeros(Ncircuit, L*T);

        ## Loading randomVecs and ProbArray of measurements:
    
        csv_reader = CSV.File("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")

        df = DataFrame(csv_reader)
        println(size(df, 1),',', size(df, 2))
    
        rv1 = zeros(size(df, 1), size(df, 2));

        println("size rv1 = ", size(rv1), size(rv2))

        for i in 1:size(rv1)[1]
            #println("i = ", i)
            for j in 1:size(rv1)[2]
                rv1[i, j] = df[i, j]
            end
        end
        println("rv1[1, 1:10] = ", rv1[1, 1:10])
        csv_reader = CSV.File("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")

        df = DataFrame(csv_reader)
        println(size(df, 1),',', size(df, 2))
    
        rv2 = zeros(size(df, 1), size(df, 2));
        for i in 1:size(rv2)[1]
            #println("i = ", i)
            for j in 1:size(rv2)[2]
                rv2[i, j] = df[i, j]
            end
        end
        save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", rv1)
        save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", rv2)                
    end
    flush(stdout)    
    

    
    @time begin
    PT = T-1
    probArr = zeros(T, 2*L)
    x = zeros(T, 2*L)
    A = zeros(1, 2*L)    

    if p==0.3
        Prob = "0p3"
    elseif p == 0.05
        Prob = "0p05"
            
    elseif p == 0.2
        Prob = "0p2"
    elseif p == 0.1
        Prob = "0p1"
    elseif p == 0.15
        Prob = "0p15"
    elseif p == 0.25
        Prob = "0p25"            
    elseif p == 0.35
        Prob = "0p35"            
    elseif p == 0.4
        Prob = "0p4"            
    end
        
        
    purifyTimeVec = zeros(Ncircuit);
        
    purifyTime = 0
    measureRes = zeros(Nsamp, TEvolve, 2*L+1);
    measureRes[:, :, :] .= NaN;
    finalMeasures = zeros(Nsamp, 3);
    
    determ = zeros(Nsamp, 3);
    fmeasureDeterm = zeros(Nsamp, 3)

    rv1 = zeros(Ncircuit, L*T);
    rv2 = zeros(Ncircuit, L*T);
    probArrMat = zeros(Ncircuit, T, 2*L);
    circuitVec = 1:Ncircuit;
    EEVec = zeros(Ncircuit)
    keepTraj = true
    ## Loading randomVecs and ProbArray of measurements:

    flush(stdout)
    if circIndVec==false 
        if Nbatch==false
            print("Nbatch==false")            
            circIndArr = load("circIndArrL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]                        
        else 
            print("Nbatch==true")            
            circIndArr = load("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.jld")["data"]
            println("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.jld")            
        end    
        circIndVec = circIndArr[1:NcircPT, PT]
    end
    
    for circInd in circIndVec #Int(size(flatCircInd)[1])   
        print("circInd = ", circInd)
        flush(stdout)    
        nb = Int(floor((circInd-1)/Ncircuit))+1

        remainCircInd = (circInd-1)%Ncircuit+1
        if Nbatch==false        
            rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
            println("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")
            rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
            println("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")
            probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
            println("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")
        else
            rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
            println("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")
            rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
            println("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")            
            probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
            println("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")        
        end
        
        println("circInd = ", circInd)
        flush(stdout)    

        @inbounds Threads.@threads for i in 1:Nsamp
            if i%1==0
                println("i = $i")
                flush(stdout)   
            end
        
            if useQCliffOrQClifford==1
                A = QCliffTimeEvolveAncilla(L, p, TEvolve, Bool(Scramble), rv1[remainCircInd, :], 
                    rv2[remainCircInd, :], probArrMat[remainCircInd, :, :]);
            else
                A = QCliffordAncilla(L, p, TEvolve, Bool(Scramble), keepTraj, rv1[remainCircInd, :], 
                    rv2[remainCircInd, :], probArrMat[remainCircInd, :, :])            
            end
                
            finalMeasures[i, :] = A[3][:]
            determ[i, :] = A[4][:]

            for j = 1:3
                fmeasureDeterm[i, j] = finalMeasures[i, j] * determ[i, j]
            end
            #println("end i = ", i)
            #println("A[3] = ", fmeasureDeterm[i, :])
            print("size A2 = ", size(A[2][:, :]   ))
            print("size measureRes = ", size(measureRes))                
            measureRes[i, :, :] = A[2][:, :]   
            purifyTime = A[5]
            println("purifyTime = ", purifyTime)        
        end
        
        swapMeasureRes = permutedims(measureRes, [1, 3, 2])
        #printMatrix(swapMeasureRes[1, :, :])
        if severalJob==false
            if T==4*L && Nbatch==false && v==false
                outfile = "measureL$L\aN$Nsamp\aP$Prob\aCrcInd$circInd$ScrambleLabel$ScrambleLabel.txt"
                outfile = "measureL$L\aNT$T\a$Nsamp\aP$Prob\aCrcInd$circInd$ScrambleLabel$ScrambleLabel.txt"
                    
            elseif T!=4*L && Nbatch==false && v==false
                outfile = "measureL$L\aT$T\aN$Nsamp\aP$Prob\aCrcInd$circInd$ScrambleLabel$ScrambleLabel.txt"
                    
            elseif T!=4*L && Nbatch!=false && v==false
                outfile = "measureL$L\aT$T\aN$Nsamp\aP$Prob\aCrcInd$circInd\aNbatch$Nbatch$ScrambleLabel$ScrambleLabel.txt"
            elseif T!=4*L && Nbatch==false && v!=false
                outfile = "measureL$L\aT$T\aN$Nsamp\aP$Prob\aCrcInd$circInd\av$v$ScrambleLabel$ScrambleLabel.txt"
            elseif T!=4*L && Nbatch!=false && v!=false
                outfile = "measureL$L\aT$T\aN$Nsamp\aP$Prob\aCrcInd$circInd\aNbatch$Nbatch\av$v$ScrambleLabel$ScrambleLabel.txt"
            end
        elseif severalJob==true
            if T==4*L && Nbatch==false && v==false
                outfile = "measureL$L\aN$Nsamp\aP$Prob\aCrcInd$circInd\aJ$job_id$ScrambleLabel$ScrambleLabel.txt"
                outfile = "measureL$L\aT$T\aN$Nsamp\aP$Prob\aCrcInd$circInd\aJ$job_id$ScrambleLabel$ScrambleLabel.txt"                    
            elseif T!=4*L && Nbatch==false && v==false
                outfile = "measureL$L\aT$T\aN$Nsamp\aP$Prob\aCrcInd$circInd\aJ$job_id$ScrambleLabel$ScrambleLabel.txt"
            elseif T!=4*L && Nbatch!=false && v==false
                outfile = "measureL$L\aT$T\aN$Nsamp\aP$Prob\aCrcInd$circInd\aNbatch$Nbatch\aJ$job_id$ScrambleLabel$ScrambleLabel.txt"
            end
        end
        
        println("outfile = ", outfile)

        flush(stdout)
    
        open(outfile, "w") do f   
            n = 1    
            #println("n = $n")
            while n <= size(swapMeasureRes)[1]
                sizeI = size(swapMeasureRes[n, :, :])
                #n = 1
                for t in 1:TEvolve
                    tempI = 0
                    for i in swapMeasureRes[n, 1:2*L, t]               

                        if i == 1.0
                            tempI = 1 #S_z = -1
                        elseif isnan(i)
                            tempI = 0     # No measurement           
                        elseif i == 2.0
                            tempI = 2  # S_z = +1
                            #print("Sz")
                        end
                        i = tempI
                        #println("tempI = $tempI")                
                        print(f, i)
                        print(f, ' ')  
                    end
                end
                for i in fmeasureDeterm[n, 1:3] # This prints the final ancilla qbit state to the end of each monitoring 
                    #println("i = ", i)
                    tempI = 0
                    if i == 1.0
                        tempI = 1 #S_z = -1
                    elseif isnan(i)
                        tempI = 0     # No measurement           
                    elseif i == 2.0
                        tempI = 2  # S_z = +1
                    end
                    i = tempI
                    print(f, Int(i))
                    print(f, ' ') 
                end            
                #println(n)        
                #println("purifyTimeVec[1] = ", purifyTimeVec[1])
                #println("purifyTime2 = ", purifyTime)   
                print(f, Int(purifyTime))
                #print(f, Int(purifyTimeVec[remainCircInd]))
                print(f, "\n")  
                n = n + 1
            end
        end
    end
    end
end

#A = determinePurifyTimeARGS(ARGS)
#A = circIndArrGenerationARGS(ARGS)


function createMeasureRec(L, p, T, Scramble, Nsamp, Ncircuit, Nbatch, NpureT, NcircPT, TEvolve, circIndVec=false, csv=false, v=false)
    if TEvolve==Any[]
        TEvolve = T
    end
    
    println("TEvolve = ", TEvolve)
    #Scramble = parse(Int, args[8])    
    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 


    if !Bool(Scramble)
        ScrambleLabel = ""
    elseif Bool(Scramble)
        ScrambleLabel = "Scrambled"
    end 
    
    keepTraj = true
    
    if csv
        rv1 = zeros(Ncircuit, L*T);
        rv2 = zeros(Ncircuit, L*T);

        ## Loading randomVecs and ProbArray of measurements:
    
        csv_reader = CSV.File("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")
        df = DataFrame(csv_reader)
        println(size(df, 1),',', size(df, 2))
        rv1 = zeros(size(df, 1), size(df, 2));
        for i in 1:size(rv1)[1]
            #println("i = ", i)
            for j in 1:size(rv1)[2]
                rv1[i, j] = df[i, j]
            end
        end
        csv_reader = CSV.File("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")

        df = DataFrame(csv_reader)
        println(size(df, 1),',', size(df, 2))
    
        rv2 = zeros(size(df, 1), size(df, 2));
        for i in 1:size(rv2)[1]
            #println("i = ", i)        
            for j in 1:size(rv2)[2]
                rv2[i, j] = df[i, j]
            end
        end
        save("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", rv1)
        save("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld", "data", rv2)                
    end
    flush(stdout)
    if circIndVec==false
        PT = T-1            
    else
        PT = 1
    end
    
    probArr = zeros(T, 2*L)
    x = zeros(T, 2*L)
    A = zeros(1, 2*L)    
    
    if p==0.3
        Prob = "P0p3"
    elseif p == 0.05
        Prob = "P0p05"            
    elseif p == 0.2
        Prob = "P0p2"
    elseif p == 0.1
        Prob = "P0p1"
    elseif p == 0.15
        Prob = "P0p15"
    elseif p == 0.25
        Prob = "P0p25"            
    elseif p == 0.35
        Prob = "P0p35"            
    elseif p == 0.4
        Prob = "P0p4"            
    end
    
    purifyTimeVec = zeros(Ncircuit);
    measureRes = zeros(Nsamp, TEvolve, 2*L+1);
    measureRes[:, :, :] .= NaN;
    finalMeasures = zeros(Nsamp, 3);
    
    determ = zeros(Nsamp, 3);
    fmeasureDeterm = zeros(Nsamp, 3)

    rv1 = zeros(Ncircuit, L*T);
    rv2 = zeros(Ncircuit, L*T);
    probArrMat = zeros(Ncircuit, T, 2*L);
    circuitVec = 1:Ncircuit;
    EEVec = zeros(Ncircuit)
    
    ## Loading randomVecs and ProbArray of measurements:
    flush(stdout)
    #csv_reader = CSV.File("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.csv")    
    
    if Nbatch==false || Nbatch==1 
        try             
            circIndArr = load("circIndArrL$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")["data"]                            
        catch y                
            if isa(y, ArgumentError)
                circIndArr = load("circIndArrL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]                
            end
        end        
    else
        circIndArr = load("circIndArrL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch$ScrambleLabel.jld")["data"]
    end
    
    print("circIndArr = ", circIndArr)
    #for j in 1:10
    #    for i in 1:20
    #        print(circIndArr[i, j])
    #        print(',')
    #    end
    #    print("\n")
    #end
    #print("PT =", PT)
    #println("circIndArr[:, PT] = ", circIndArr[:, PT])
    
    if circIndVec==false 
        circIndVec = circIndArr[1:NcircPT, PT]
    end
    println("circIndVec in createMeasureRec = ", circIndVec)
    
    for circInd in circIndVec #Int(size(flatCircInd)[1])
        if circInd==0
            continue
        end
        #nb = Int(floor(circInd/Ncircuit))+1
        nb = Int(floor((circInd-1)/Ncircuit))+1
        
        remainCircInd = (circInd-1)%Ncircuit+1

        println("remainCircInd = ", remainCircInd)
        
        if Nbatch == false || Nbatch == 1
            try 
                println("try 1")
                println("randVec1L$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")
                rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")["data"]
                rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")["data"]
                probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p$ScrambleLabel.jld")["data"]                
                
            catch y                
                if isa(y, ArgumentError)
                    rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
                    rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]
                    probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T$ScrambleLabel.jld")["data"]                
                end
            end
        else      
            try
                rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
                rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
                probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]                
                println("probArrMatL$L-Ncirc$Ncircuit-p$p-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")                                            
            catch y                
                if isa(y, ArgumentError)                
                    rv1 = load("randVec1L$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
                    rv2 = load("randVec2L$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]
                    probArrMat = load("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")["data"]                
                    println("probArrMatL$L-Ncirc$Ncircuit-p$p-T$T-Nbatch$Nbatch-Nb$nb$ScrambleLabel.jld")                            
                end                
            end
        end
        
        println("circInd = ", circInd, " remainCircInd = ", remainCircInd, " nb = ", nb)
        flush(stdout)
        purifyTime = 0
        Threads.@threads for i in 1:Nsamp
            if i%100==0
                println("i = $i")
            end
            #if i%100==0
            #    println("i = $i")
            #end
            flush(stdout)
            if useQCliffOrQClifford==1
                A = QCliffTimeEvolveAncilla(L, p, TEvolve, Bool(Scramble), rv1[remainCircInd, :], 
                    rv2[remainCircInd, :], probArrMat[remainCircInd, :, :]);
            else
                A = QCliffordAncilla(L, p, TEvolve, Bool(Scramble), keepTraj, rv1[remainCircInd, :], 
                    rv2[remainCircInd, :], probArrMat[remainCircInd, :, :])            
            end
        
            finalMeasures[i, :] = A[3][:]
            determ[i, :] = A[4][:]
            purifyTimeVec[i] = A[5]
            flush(stdout)
            for j = 1:3
                fmeasureDeterm[i, j] = finalMeasures[i, j] * determ[i, j]
            end
            measureRes[i, :, :] = A[2][:, :]
            purifyTime = A[5]
            if i==1
                println("purifyTime = ", purifyTime)
            end
            flush(stdout)                
        end
        swapMeasureRes = permutedims(measureRes, [1, 3, 2])
        flush(stdout)
        if T==4*L && Nbatch==false
            outfile = "measureL$L\aN$Nsamp\a$Prob\aCrcInd$circInd$ScrambleLabel.txt"
        elseif T!=4*L && Nbatch==false
            outfile = "measureL$L\aT$T\aN$Nsamp\a$Prob\aCrcInd$circInd$ScrambleLabel.txt"
        elseif T!=4*L && Nbatch!=false
            outfile = "measureL$L\aT$T\aN$Nsamp\a$Prob\aCrcInd$circInd\aNbatch$Nbatch$ScrambleLabel.txt"
        end
        println("outfile = ", outfile)
    
        open(outfile, "w") do f   
            n = 1    
            #println("n = $n")
            while n <= size(swapMeasureRes)[1]
                sizeI = size(swapMeasureRes[n, :, :])
                #n = 1
                for t in 1:TEvolve
                    tempI = 0
                    #println("swapMeasureRes[n, 1:2*L, t] = ", swapMeasureRes[n, 1:2*L, t])
                    for i in swapMeasureRes[n, 1:2*L, t]               

                        if i == 1.0
                            tempI = 1 #S_z = -1
                        elseif isnan(i)
                            tempI = 0     # No measurement   
                        elseif i == 2.0
                            tempI = 2  # S_z = +1
                        end
                        i = tempI
                        print(f, i)
                        print(f, ' ')
                    end
                end
                for k in fmeasureDeterm[n, 1:3] # This prints the final ancilla qbit state to the end of each monitoring 
                    println("fmeasureDeterm[k, j] = ", k)                        
                    tempI = 0
                        
                    if k == 1.0
                        tempI = 1 #S_z = -1
                    elseif isnan(k)
                        tempI = 0     # No measurement           
                    elseif k == 2.0
                        tempI = 2  # S_z = +1
                    end
                    k = tempI
                    println("k, tempI = ", k, tempI)
                    print(f, Int(k))
                    print(f, ' ') 
                end  
                println("end of fmeasureDeterm")
                print(f, Int(purifyTime))
                print(f, "\n")  
                flush(stdout)                    
                n = n + 1
            end    
        end
    end
end

function combinedSampCreator(L, p, T, Scramble, Ncircuit, Nbatch, NpureT, NcircPT)
    A = randArrGeneration(L, p, T, Ncircuit, Nbatch, Scramble)
    circInd = false;nb=false;csv=false;
    A = determinePurifyTime(L, p, T, Scramble, Ncircuit, Nbatch, nb, circInd, csv);
    B = circIndArrGeneration(L, p, T, Scramble, Ncircuit, Nbatch, NpureT, NcircPT);
    #A = determinePurifyTime(L, p, T, Scramble, Ncircuit, Nbatch=false, nb=false, cVec = false, csv=false)
end

A = createMeasureRecArg(ARGS)

#export JULIA_NUM_THREADS=16
