#include "FCT.h"

#include <iomanip>
#include<time.h>

// 3-D
/*
int main()
{
    FCT<3> fct;
    int mesh[3];
    int Nx = 4,
        Ny = 4,
        Nz = 8;
    int size = Nx * Ny * Nz;
    mesh[0] = Nx;
    mesh[1] = Ny;
    mesh[2] = Nz;
    double *data_c, *data;
    data_c = new double[size];
    cudaMalloc((void**)&data, size * sizeof(double));

    srand((unsigned int)time(NULL));
    for(int i = 0; i < size; ++i)
        // data_c[i] = 1.0;
        data_c[i] = rand()%10;
    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                std::cout << std::setw(8) << std::scientific
                << data_c[z + y*Nz + Nz*Ny*x] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "----------------------------------------------------------------" << std::endl;

    cudaMemcpy(data, data_c, size * sizeof(double), cudaMemcpyHostToDevice);

    fct.setup(mesh);
    fct.forwardTransform(data);

    cudaMemcpy(data_c, data, size * sizeof(double), cudaMemcpyDeviceToHost);

    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                std::cout << std::setw(8) << std::scientific
                << data_c[z + y*Nz + Nz*Ny*x] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "----------------------------------------------------------------" << std::endl;
    fct.inverseTransform(data);

    cudaMemcpy(data_c, data, size * sizeof(double), cudaMemcpyDeviceToHost);

    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                std::cout << std::setw(8) << std::scientific
                << data_c[z + y*Nz + Nz*Ny*x] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    free(data_c);
    cudaFree(data);

    return 0;
}
*/

// 2-D
/*int main()
{
    int mesh[3], perm[3];
    int Nz = 6,
        Ny = 4;
    int size = Nz * Ny;
    mesh[0] = 1;
    mesh[1] = Ny;
    mesh[2] = Nz;

    perm[0] = 1;
    perm[1] = 0;
    perm[2] = 2;

    cufftDoubleReal *in_d, *out_d, *in_c, *out_c;
    in_c  = new cufftDoubleReal[size];
    out_c = new cufftDoubleReal[size];
    cudaMalloc((void**)&in_d,  size * sizeof(cufftDoubleReal));
    cudaMalloc((void**)&out_d, size * sizeof(cufftDoubleReal));

    srand((unsigned int)time(NULL));
    for(int i = 0; i < size; ++i)
        in_c[i] = rand()%10;
    
    for(int y = 0; y < Ny; ++y)
    {
        for(int z = 0; z < Nz; ++z)
            std::cout << in_c[z + Nz*y] << "   ";
        std::cout << "\n";
    }
    std::cout << "\n";

    cudaMemcpy(in_d, in_c, size*sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);

    cut_transpose3d( out_d,
                     in_d,
                     mesh,
                     perm,
                     1);
    cudaMemcpy(out_c, out_d, size*sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);    

    for(int y = 0; y < Nz; ++y)
    {
        for(int z = 0; z < Ny; ++z)
        {
            std::cout << out_c[y+Nz*z] << "   ";
            // std::cout << y+Ny*x+1 << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    mesh[0] = 1;
    mesh[1] = Nz;
    mesh[2] = Ny;

    perm[0] = 2;
    perm[1] = 1;
    perm[2] = 0;

    cut_transpose3d( in_d,
                     out_d,
                     mesh,
                     perm,
                     1);
    cudaMemcpy(out_c, in_d, size*sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);    

    for(int y = 0; y < Ny; ++y)
    {
        for(int z = 0; z < Nz; ++z)
        {
            std::cout << out_c[y+Ny*z] << "   ";
            // std::cout << y+Ny*x+1 << "\n";
        }
        std::cout << "\n";
    }

    delete [] in_c; 
    delete [] out_c;
    cudaFree(in_d);
    cudaFree(out_d);


   FCT<2> fct;
    int mesh[3];
    int Nx = 1,
        Ny = 4,
        Nz = 4;
    int size = Nx * Ny * Nz;
    mesh[0] = Nx;
    mesh[1] = Ny;
    mesh[2] = Nz;
    double *data_c, *data;
    data_c = new double[size];
    cudaMalloc((void**)&data, size * sizeof(double));

    srand((unsigned int)time(NULL));
    for(int i = 0; i < size; ++i)
        // data_c[i] = 1.0;
        data_c[i] = rand()%10;
    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                std::cout << std::setw(8) << std::scientific
                          << data_c[z + y*Nz + Nz*Ny*x] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "----------------------------------------------------------------" << std::endl;

    cudaMemcpy(data, data_c, size * sizeof(double), cudaMemcpyHostToDevice);

    fct.setup(mesh);
    fct.forwardTransform(data);
    // exit(1);

    cudaMemcpy(data_c, data, size * sizeof(double), cudaMemcpyDeviceToHost);

    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                std::cout << std::setw(8) << std::scientific
                << data_c[z + y*Nz + Nz*Ny*x] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "----------------------------------------------------------------" << std::endl;
    fct.inverseTransform(data);

    cudaMemcpy(data_c, data, size * sizeof(double), cudaMemcpyDeviceToHost);

    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                std::cout << std::setw(8) << std::scientific
                << data_c[z + y*Nz + Nz*Ny*x] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    free(data_c);
    cudaFree(data);

    return 0;

}*/

int main()
{
    FCT<1> fct;
    int mesh[1];
    int Nx = 1,
        Ny = 1,
        Nz = 16;
    int size = Nx * Ny * Nz;
    mesh[0] = Nx;
    mesh[1] = Ny;
    mesh[2] = Nz;
    double *data_c, *data;
    data_c = new double[size];
    cudaMalloc((void**)&data, size * sizeof(double));

    srand((unsigned int)time(NULL));
    for(int i = 0; i < size; ++i)
        data_c[i] = std::cos(M_PI*(i+0.5)/size);
    
        // data_c[i] = rand()%10;
    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                std::cout << std::setw(8) << std::scientific
                << data_c[z + y*Nz + Nz*Ny*x] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "----------------------------------------------------------------" << std::endl;

    cudaMemcpy(data, data_c, size * sizeof(double), cudaMemcpyHostToDevice);

    fct.setup(mesh);
    fct.forwardTransform(data);

    cudaMemcpy(data_c, data, size * sizeof(double), cudaMemcpyDeviceToHost);

    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                std::cout << std::setw(8) << std::scientific
                << data_c[z + y*Nz + Nz*Ny*x] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "----------------------------------------------------------------" << std::endl;
    fct.inverseTransform(data);

    cudaMemcpy(data_c, data, size * sizeof(double), cudaMemcpyDeviceToHost);

    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                std::cout << std::setw(8) << std::scientific
                << data_c[z + y*Nz + Nz*Ny*x] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    free(data_c);
    cudaFree(data);

    return 0;
}
