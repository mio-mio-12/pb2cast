// Legacy native oracle only; production uses animation_source.py.
// This client DLL rounds time/step before truncating, but computes alpha
// with a separate fmod. At boundaries those select inconsistent intervals.
// The final bool is unused in the inspected DLL; toggling it is not a fix.
#include <windows.h>
#include <cstdio>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstddef>
#include <iomanip>
// Verified offsets in this x86 client build; unreferenced fields remain opaque.
struct I3MatrixInfo { float time; unsigned opaque04[6]; unsigned playbackMode; unsigned opaque20; float* translation; float* rotation; float* scale; unsigned opaque30[4]; };
static_assert(sizeof(I3MatrixInfo)==0x40);
static_assert(offsetof(I3MatrixInfo,playbackMode)==0x1c);
static_assert(offsetof(I3MatrixInfo,translation)==0x24);
static_assert(offsetof(I3MatrixInfo,rotation)==0x28);
static_assert(offsetof(I3MatrixInfo,scale)==0x2c);
static void* sym(HMODULE h,const char*n){auto p=GetProcAddress(h,n);if(!p)throw std::runtime_error(n);return p;}
static std::string quote(const std::string&s){std::string r="\"";for(char c:s){if(c=='\\'||c=='\"')r+='\\';if((unsigned char)c>=32)r+=c;}return r+"\"";}
int main(int argc,char**argv){try{
 SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);if(argc!=4)return 2;SetDllDirectoryA(argv[1]);auto h=LoadLibraryA((std::string(argv[1])+"\\i3MathDx.dll").c_str());if(!h)throw std::runtime_error("Engine library unavailable");auto hb=GetModuleHandleA("i3BaseDx.dll");
 auto mgr=((void*(__cdecl*)())sym(h,"?new_object_fun@i3AnimationResManager@@SAPAV1@XZ"))();auto obj=((void*(__cdecl*)())sym(h,"?new_object_fun@i3AnimationPackFile@@SAPAV1@XZ"))();
 auto load=(unsigned(__thiscall*)(void*,char*))sym(h,"?LoadFromFile@i3AnimationPackFile@@QAEIPAD@Z");auto find=(void*(__thiscall*)(void*,const char*))sym(hb,"?FindResourceA@i3NamedResourceManager@@QAEPAVi3ResourceObject@@PBD@Z");
 auto count=(int(__thiscall*)(void*))sym(h,"?GetTrackCount@i3Animation@@QAEHXZ");auto duration=(float(__thiscall*)(void*))sym(h,"?GetDuration@i3Animation@@QAEMXZ");auto name=(char*(__thiscall*)(void*,int))sym(h,"?GetTrackBoneName@i3Animation@@QAEPADH@Z");auto sample=(unsigned(__thiscall*)(void*,int,I3MatrixInfo*,bool))sym(h,"?GetInterpolatedKeyframe@i3Animation2@@UAEIHPAUI3MATRIXINFO@@_N@Z");
 std::ifstream f(argv[2],std::ios::binary);std::vector<char>d((std::istreambuf_iterator<char>(f)),{});if(d.size()<164||(memcmp(d.data(),"APF1",4) && memcmp(d.data(),"APF2",4) && memcmp(d.data(),"APF3",4)))throw std::runtime_error("Expected APF1, APF2 or APF3");unsigned nc=*(unsigned*)(d.data()+4);if(nc>10000||164+nc*284>d.size())throw std::runtime_error("Invalid APF directory");
 if(load(obj,argv[2])==~0u)throw std::runtime_error("Engine failed to load animation pack");
 std::ofstream out(argv[3]);out<<std::setprecision(9);out<<"{\"fps\":30,\"clips\":[";
 for(unsigned i=0;i<nc;i++){
  std::string path(d.data()+184+i*284,strnlen(d.data()+184+i*284,260));auto a=find(mgr,path.c_str());if(!a)throw std::runtime_error("Animation missing after load");int nt=count(a);float seconds=duration(a);if(nt<1||nt>1024||!std::isfinite(seconds)||seconds<0||seconds>600)throw std::runtime_error("Invalid track count or duration");int frames=(int)ceil(seconds*30)+1;
  if(i)out<<",";out<<"{\"name\":"<<quote(path)<<",\"duration\":"<<seconds<<",\"tracks\":[";
  for(int t=0;t<nt;t++){if(t)out<<",";out<<"{\"name\":"<<quote(name(a,t))<<",\"samples\":[";
   unsigned trackFlags=0;for(int fr=0;fr<frames;fr++){
    I3MatrixInfo info{};float pos[3]={},rot[4]={0,0,0,1},scale[3]={1,1,1};info.time=(std::min)(seconds,fr/30.0f);info.translation=pos;info.rotation=rot;info.scale=scale;unsigned flags=sample(a,t,&info,false);trackFlags|=flags;
    if(fr)out<<",";out<<"[";float values[10]={pos[0],pos[1],pos[2],rot[0],rot[1],rot[2],rot[3],scale[0],scale[1],scale[2]};for(int k=0;k<10;k++){if(!std::isfinite(values[k]))throw std::runtime_error("Non-finite animation sample");if(k)out<<",";out<<values[k];}out<<"]";
   }out<<"],\"flags\":"<<trackFlags<<"}";
  }out<<"]}";
 }out<<"]}";printf("Decoded %u clips at 30 fps using the client animation library.\n",nc);return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}

