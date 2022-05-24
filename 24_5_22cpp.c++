#include <iostream>

using namespace std;

class Distance {
	
	private:
		int meters;
	public:
		Distance()
		{
			meters = 0;
		}
		void displayData()
		{
			cout << "Meters value : " << meters;
		}
		friend void addValue(Distance& d);
};
void addValue(Distance& d)
{
	d.meters = d.meters + 5;
}

class Shape {

	protected:
		int width;
		int height;
	public:
		void setWidth(int w) {
			width = w;
		}
		void setHeight(int h) {
			height = h;
		}

};
class Rectangle : public Shape {
		
	public:
		int getArea() {
			return width * height
		}

};

class MyBaseClas {
	
	public:
		int x;

		MyBaseClas()
		{
			x = 0;
			y = 0;
			z = 0;
		}
		void printProtectedData()
		{
			cout << "Y: " << y << endl;
		}

	protected:
		int y;
	private:
		int z;
};

class MyDerivedClass : public MyBaseClas {

};

void MyOutssideFunction(MyBaseClas obj)
{
	cout << "X: " << obj.x << endl;
	//obj.printProtectedData();
	//cout << "Z: " << obj.z << endl;
}



int main()
{
	/*Distance d1;
	d1.displayData();
	addValue(d1);
	d1.displayData();*/

	Rectangle rect;
	rect.setWidth(5);
	rect.setHeight(7);

	MyBaseClas obj1;
	MyDerivedClass obj2;
	MyOutssideFunction(obj1);
	cout << "X: " << obj2.x;



	return 0;
}
