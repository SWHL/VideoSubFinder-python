#include <iostream>

// #include "Components/Include/DataTypes.h"
// #include "vsf_cpp/IPAlgorithms.h"
// using namespace std;
//
// void ColorFiltration(simple_buffer<u8>& ImBGR, simple_buffer<int>& LB, simple_buffer<int>& LE, int& N, int w, int h)
// {
// 	custom_assert(g_segw > 0, "ColorFiltration: g_segw > 0");
// 	const int scd = g_scd, segw = g_segw, msegc = g_msegc, mx = (w - 1) / g_segw;
// 	s64 t1, dt, num_calls;
//
// 	custom_assert(ImBGR.size() >= w * h * 3, "ColorFiltration(...)\nnot: ImBGR.size() >= w*h*3");
// 	custom_assert(LB.size() >= h, "ColorFiltration(...)\nnot: LB.size() >= H");
// 	custom_assert(LE.size() >= h, "ColorFiltration(...)\nnot: LE.size() >= H");
//
// 	simple_buffer<int> line(h, 0);
//
// 	std::for_each(std::execution::par, ForwardIteratorForDefineRange<int>(0), ForwardIteratorForDefineRange<int>(h), [&](int y)
// 	{
// 		int ib = w * y;
// 		int cnt = 0;
// 		int i, ia, nx, mi, dif, rdif, gdif, bdif;
// 		int r0, g0, b0, r1, g1, b1;
//
// 		for (nx = 0, ia = ib; nx<mx; nx++, ia += segw)
// 		{
// 			b0 = ImBGR[(ia * 3)];
// 			g0 = ImBGR[(ia * 3) + 1];
// 			r0 = ImBGR[(ia * 3) + 2];
//
// 			mi = ia + segw;
// 			dif = 0;
//
// 			for (i = ia + 1; i <= mi; i++)
// 			{
// 				b1 = ImBGR[(i * 3)];
// 				g1 = ImBGR[(i * 3) + 1];
// 				r1 = ImBGR[(i * 3) + 2];
//
// 				rdif = r1 - r0;
// 				if (rdif<0) rdif = -rdif;
//
// 				gdif = g1 - g0;
// 				if (gdif<0) gdif = -gdif;
//
// 				bdif = b1 - b0;
// 				if (bdif<0) bdif = -bdif;
//
// 				dif += rdif + gdif + bdif;
//
// 				r0 = r1;
// 				g0 = g1;
// 				b0 = b1;
// 			}
//
// 			if (dif >= scd) cnt++;
// 			else cnt = 0;
//
// 			if (cnt == msegc)
// 			{
// 				line[y] = 1;
// 				break;
// 			}
// 		}
// 	});
//
// 	simple_buffer<int> lb(h, 0), le(h, 0);
// 	int n = 0;
// 	int sbegin = 1; //searching begin
// 	int y;
// 	for (y = 0; y<h; y++)
// 	{
// 		if (line[y] == 1)
// 		{
// 			if (sbegin == 1)
// 			{
// 				lb[n] = y;
// 				sbegin = 0;
// 			}
// 		}
// 		else
// 		{
// 			if (sbegin == 0)
// 			{
// 				le[n] = y - 1;
// 				sbegin = 1;
// 				n++;
// 			}
// 		}
// 	}
// 	if (sbegin == 0)
// 	{
// 		le[n] = y - 1;
// 		n++;
// 	}
//
// 	if (n == 0)
// 	{
// 		N = 0;
// 		return;
// 	}
//
// 	if (g_show_results) SaveBGRImageWithLinesInfo(ImBGR, "/DebugImages/ColorFiltration_01_ImBGRWithLinesInfo" + g_im_save_format, lb, le, n, w, h);
//
// 	int dd, bd, md;
//
// 	dd = 12;
// 	bd = 2 * dd;
// 	md = g_min_h * h;
//
// 	int k = 0;
// 	int val = lb[0] - dd;
// 	if (val<0) val = 0;
// 	LB[0] = val;
//
// 	for (int i = 0; i<n - 1; i++)
// 	{
// 		if ((lb[i + 1] - le[i] - 1) >= bd)
// 		{
// 			if ((le[i] - LB[k]) >= md)
// 			{
// 				LE[k] = le[i] + dd;
// 				k++;
// 				LB[k] = lb[i + 1] - dd;
// 			}
// 			else
// 			{
// 				LB[k] = lb[i + 1] - dd;
// 			}
// 		}
// 	}
// 	if ((le[n-1] - LB[k]) >= md)
// 	{
// 		val = le[n-1] + dd;
// 		if (val>h - 1) val = h - 1;
// 		LE[k] = val;
// 		k++;
// 	}
//
// 	N = k;
//
// 	if ((N > 0) && (g_show_results)) SaveBGRImageWithLinesInfo(ImBGR, "/DebugImages/ColorFiltration_02_ImBGRWithLinesInfoUpdated" + g_im_save_format, LB, LE, N, w, h);
// }
//

// int main() {
//     cout << "Hello, World!" << endl;
// 	simple_buffer<int> LB(h, 0), LE(h, 0);
// 	int N;
//
// 	ColorFiltration(ImBGR, LB, LE, N, w, h);
//
//     return 0;
// }


#include <opencv2/opencv.hpp>

using namespace cv;
void ImageThreshold(String str) {
  Mat image = imread(str);

  imshow("test_opencv",image);
  waitKey(0);
}
int main() {
  String str = "/Users/jiahuawang/projects/VideoSubFinder-python/tmp/1.jpg";
  ImageThreshold(str);
  return 0;
}