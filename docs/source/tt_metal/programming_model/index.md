Programming Model
===================

TT-Metalium is a platform for programming heterogeneous collection of CPUs (host processors) and Tenstorrent acceleration devices,
composed of many RISC-V processors. Target users of TT-Metal are expert parallel programmers wanting
to write parallel and efficient code with full access to the Tensix hardware via low-level kernel APIs.

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide2.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide3.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide4.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide5.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide6.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide7.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide8.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide9.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide10.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide11.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide12.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide13.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide14.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide15.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide16.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide17.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide18.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide19.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide20.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide21.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide22.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide23.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide24.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide25.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide26.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide27.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide28.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide29.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide30.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide31.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide32.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide33.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide34.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide35.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide36.jpeg">

<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/blob/mbahnas/vit_ttnn_L1_interleaved/docs/source/images/programming_examples/Slide37.jpeg">
