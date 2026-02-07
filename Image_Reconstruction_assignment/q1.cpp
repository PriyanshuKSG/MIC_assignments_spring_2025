#include<iostream>
// #include<simplecpp>

using namespace std;

// main_program {
// }
int main()
{
    char ch;
    cin>>ch;

    char ch2;

    int count = 1;
    int score = 0;

    for(int i = 1; i <= 7; i++)
    {
        cin>>ch2;
        //cout<<ch<<" "<<ch2<<endl;
        if(ch2 == ch)
        {
            count++;
        }
        else{
            score += 2*(count/4);
            //cout<<score<<endl;
            score += 3*(count/3);
            //cout<<score<<endl;
            count = 1;
        }

        //cout<<score<<endl;
        ch = ch2;
    }

    //cout<<count<<endl;
    score += 2*(count/4);
    score += 3*(count/3);
    cout<<score<<endl;

    return 0;
}