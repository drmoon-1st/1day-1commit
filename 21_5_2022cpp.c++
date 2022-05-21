#include <iostream>
#include <string>

using namespace std;

struct Person {

	string name;
	int age;
	double salary;

};

class cars {

	private:
	string conpany_name;
	string modle_name;
	string fuel_type;
	float mileage;
	double price;

	public:
		void setData(string cname, string mname, string ftype, float m, double p)
		{
			conpany_name = cname;
			modle_name = mname;
			fuel_type = ftype;
			mileage = m;
			price = p;
		}
		void displayData()
		{
			cout <<	data
		}
};



int main()
{
	/*Person p1;
	cout << "Enter Person details" << endl;
	cin.getline(p1.name, 100);
	cout << "Enter age" << endl;
	cin >> p1.age;
	cout << "Enter salary" << endl;
	cin >> p1.salary;

	cout << endl << "Person details are as follows : " << endl;
	cout << "Pers on name : " << p1.name << endl;
	cout << "Person age: " << p1.age << endl;
	cout << "Person salary : " << p1.salary << endl;*/

	Person p1[2];
	cout << "Enter Person details" << endl;
	
	for (int i = 0; i < 2; i++)
	{
		cin >> p1[i].name;
		cin >> p1[i].age;
		cin >> p1[i].salary;
	}
	cout << endl << "Person details are as follows : " << endl;
	for (int i = 0; i < 2; i++)
	{
		cout << "Pers on name : " << p1[i].name << endl;
		cout << "Person age: " << p1[i].age << endl;
		cout << "Person salary : " << p1[i].salary << endl;
	/*}*/
	
	cars car;
	car.setData("name", "al", "gg", 19.2, 2000.33);
	car.displayData();



	return 0;
}
