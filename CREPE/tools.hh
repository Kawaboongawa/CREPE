#pragma once
# include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>



/*! \brief Getter on max threads in one dimension
**
** Fetch the maximum number of threads available in one dimension
** for a kernel/CUDA computation. It asks directly the
** NVIDIA graphic card. This function, when called several times,
** will only ask once the hardware.
*/
unsigned int get_max_threads();

/*! \brief Getter on max threads in two dimensions
**
** Fetch the maximum number of threads available in two dimensions
** for a kernel/CUDA computation. It asks directly the
** NVIDIA graphic card. This function, when called several times,
** will only ask once the hardware.
*/
unsigned int get_max_threads_2d();

/*! \brief Getter on max blocks
**
** Fetch the maximum number of blocks available in one dimension
** for a kernel/CUDA computation. It asks directly the
** NVIDIA graphic card. This function, when called several times,
** will only ask once the hardware.
*/

unsigned int get_max_blocks();

/*! \function Given a problem of *size* elements, compute the lowest number of
* blocks needed to fill a compute grid.
*
* \param nb_threads Number of threads per block. */
unsigned map_blocks_to_problem(const size_t problem_size,
	const unsigned nb_threads);