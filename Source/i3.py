from pathlib import Path
import struct,collections
ROOT=Path(r'D:\Game\PBMVM\Point Blank V3.24 Client\PB-BAKBAKAN')
def u(d,o):return struct.unpack_from('<I',d,o)[0]
def decrypt(d,s):return bytes(((d[i-1]<<(8-s))|(d[i]>>s))&255 for i in range(len(d)))
def span(d,o,n):
 if o<0 or n<0 or o+n>len(d):raise ValueError('Invalid data range')
 return d[o:o+n]
from studio_i3 import parse_i3r2, parse_pack_entries, biah_pack_entries

def archive_entries(path, data=None):
 data=Path(path).read_bytes() if data is None else data
 if data.startswith(b'Biah'):return list(biah_pack_entries(data,str(path)))
 root=parse_i3r2(data,str(path))
 return [e for b in root.blocks_of_type('i3PackNode') for e in parse_pack_entries(b.data,root.data,str(path))]

def blocks(d):
 return {b.block_id:(b.type_name,b.data) for b in parse_i3r2(d).blocks}

def pack(d):
 for e in archive_entries('<memory>',d):yield e.name,e.data
if __name__=='__main__':
 for p in list((ROOT/'Pack').glob('Weapon_M-7*.i3Pack'))[:1]+list((ROOT/'Pack').glob('Weapon_K2.i3Pack'))[:1]:
  print(p.name)
  for name,d in pack(p.read_bytes()):
   print(name,len(d))
   if name.lower().endswith(('.i3s','.i3a')):
    try:
     bl=blocks(d);print(collections.Counter(k for k,b in bl.values()));Path('work/castdev/'+name.replace('/','_').replace('\\','_')).write_bytes(d)
    except Exception as e:print(e)
