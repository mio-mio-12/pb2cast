"""pb2cast offline asset reader. Reads client files; writes only the requested cache/export."""
import copy
import sys, os, json, struct, math, re, hashlib, subprocess, traceback, shutil, io
from pathlib import Path
import numpy as np
from PIL import Image
from i3 import blocks, pack, span, u
from official_cast import Cast
from studio_i3 import parse_i3r2, _parse_attr_set, _texture_block_name
from studio_metadata import load_weapon_metadata_index
from animation_source import decode_pack, DECODER_REV, sample_count, load_i3animpack

VERSION = 9
PIPELINE_REV = 13
MODEL_REV = 3
HOME = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
INDEX_HOME=(HOME/'../../cache/index').resolve() if getattr(sys,'frozen',False) else HOME/'index-cache'
def persistent_index(kind,stamp,build):
    key=hashlib.sha256(json.dumps([VERSION,stamp],default=str).encode()).hexdigest()
    path=INDEX_HOME/(kind+'-'+key+'.json')
    if path.exists():
        try:return json.loads(path.read_text())
        except (ValueError,OSError):pass
    value=build();save_json(path,value);return value

def clean(s): return re.sub(r'[^\w. -]+', '_', s).strip(' .') or 'asset'
def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False), encoding='utf-8'); os.replace(temp,path)
def short(d):
    n=0; shift=0; pos=0
    while True:
        x=d[pos]; pos+=1; n|=(x&127)<<shift
        if not x&128: break
        shift+=7
        if shift>28: raise ValueError('Invalid name')
    return span(d,pos,n).decode('cp1252'),pos+n
def quat(m):
    # Stable rotation-matrix conversion, column-vector convention.
    a=np.array(m,dtype=float); a=a/np.linalg.norm(a,axis=0)
    trace=np.trace(a)
    if trace>0:
        s=math.sqrt(trace+1)*2; q=[(a[2,1]-a[1,2])/s,(a[0,2]-a[2,0])/s,(a[1,0]-a[0,1])/s,s/4]
    else:
        i=int(np.argmax(np.diag(a))); j=(i+1)%3; k=(i+2)%3
        s=math.sqrt(max(0,1+a[i,i]-a[j,j]-a[k,k]))*2
        q=[0,0,0,0]; q[i]=s/4; q[j]=(a[j,i]+a[i,j])/s; q[k]=(a[k,i]+a[i,k])/s; q[3]=(a[k,j]-a[j,k])/s
    q=np.array(q); return (q/np.linalg.norm(q)).tolist()
def trs(m): return [*m[:3,3].tolist(),*quat(m[:3,:3]),*np.linalg.norm(m[:3,:3],axis=0).tolist()]
def extrefs(d):
    r=parse_i3r2(d)
    refs={b.block_id:r.text_lines[b.target] for b in r.blocks if 0 < b.target < len(r.text_lines)}
    for b in r.blocks:
        if b.type_name in ('i3Texture','i3TextureObject'):
            name=_texture_block_name(r,b)
            if name.lower().endswith(('.i3i','.dds')):refs[b.block_id]=name
    return refs

def texture(d,path):
    if d[:4]!=b'I3IB': raise ValueError('Unsupported texture header')
    w,h=struct.unpack_from('<HH',d,6); mip=struct.unpack_from('<H',d,24)[0]; body=span(d,60+d[26],len(d)-60-d[26])
    fmt={0x81:b'DXT1',0x80:b'DXT1',2:b'DXT3',4:b'DXT5'}.get(d[10])
    if not fmt or d[13] not in (0x80,0xa0): raise ValueError('Unsupported texture pixel format')
    header=[124,0xA1007,h,w,len(body),0,max(1,mip)]+[0]*11+[32,4,struct.unpack('<I',fmt)[0],0,0,0,0,0,0x401008 if mip>1 else 0x1000,0,0,0,0]
    dds=b'DDS '+struct.pack('<31I',*header)+body
    im=Image.open(io.BytesIO(dds)).convert('RGBA'); im.save(path)
    # Straight RGBA for the native preview; PNG is used by CAST materials.
    Path(str(path)+'.rgba').write_bytes(struct.pack('<II',w,h)+im.tobytes())

SHARED_ASSETS={}
def shared_asset(root,ref,cache):
    rel=Path(ref.replace('\\','/'))
    local=root/rel
    if local.is_file():return local.read_bytes()
    folder=rel.parent.as_posix().replace('/','_')
    key=(str(root),folder.lower())
    if key not in SHARED_ASSETS:
        found={}
        for p in sorted((root/'Pack').glob(folder+'*.i3Pack')):
            try:
                for n,d in pack(p.read_bytes()):found.setdefault(Path(n).name.lower(),d)
            except (ValueError,struct.error):continue
        SHARED_ASSETS[key]=found
    value=SHARED_ASSETS[key].get(rel.name.lower())
    if value is not None:return value
    indexkey=(str(root),'all')
    if indexkey not in SHARED_ASSETS:
        indexpath=cache.parent/'package-assets-index.json'
        inventory=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted((root/'Pack').glob('*.i3Pack'))]
        stamp=hashlib.sha1((str(VERSION)+json.dumps(inventory)).encode()).hexdigest()
        saved=json.loads(indexpath.read_text()) if indexpath.exists() else {}
        if saved.get('stamp')!=stamp:
            entries={}
            for name,_,_ in inventory:
                try:
                    for n,d in pack((root/'Pack'/name).read_bytes()):entries.setdefault(Path(n).name.lower(),[]).append(name)
                except (ValueError,struct.error,AssertionError):continue
            saved=dict(stamp=stamp,entries=entries);save_json(indexpath,saved)
        SHARED_ASSETS[indexkey]=saved['entries']
    for package in SHARED_ASSETS[indexkey].get(rel.name.lower(),[]):
        value=package_assets(root/'Pack'/package).get(rel.name.lower())
        if value is not None:SHARED_ASSETS[key][rel.name.lower()]=value;return value
    return None

def read_character(data,name,assets,root,cache,arms=False,player=False):
    key=hashlib.sha256(str((VERSION,MODEL_REV)).encode()+str(root).encode()+data+str((arms,player)).encode()).hexdigest()
    shared=INDEX_HOME/'characters'/key;file=shared/'model.json';shared.mkdir(parents=True,exist_ok=True)
    if file.exists():model=json.loads(file.read_text())
    else:
        model=read_model(data,name,assets,root,shared,arms=arms,player=player);save_json(file,model)
    cache.mkdir(parents=True,exist_ok=True)
    for material in model['materials'].values():
        for texture in material.values():
            for suffix in ('','.rgba'):
                source=shared/(texture+suffix);target=cache/(texture+suffix)
                if source.exists() and not target.exists():shutil.copyfile(source,target)
    return model

def remap_draw_skin(ga, blocks_by_id, indices, positions, normals, uvs, ids, weights, bone_count):
    # SGA1 stores draw-local palettes: each 32-byte record has a start index,
    # triangle count, palette length and twenty byte-sized skeleton indices.
    # The memory-buffer reference belongs to this geometry, not a nearby LOD.
    offset=25 if ga[:4]==b'GEO2' else 21
    if ga[offset:offset+4]!=b'SGA1':
        return indices,positions,normals,uvs,ids,weights
    section_count=u(ga,offset+4);buffer_id=u(ga,offset+8)
    kind,data=blocks_by_id[buffer_id]
    if kind!='i3MemoryBuffer' or len(data)!=4+section_count*32 or u(data,0)!=section_count*32:
        raise ValueError('Invalid draw skin palette buffer')
    covered=np.zeros(len(indices),dtype=bool);out_indices=np.empty(len(indices),dtype=int)
    vertices=[];out_ids=[];out_weights=[]
    for section in range(section_count):
        off=4+section*32;start=u(data,off);end=start+u(data,off+4)*3;n=u(data,off+8)
        if not 1<=n<=20 or start%3 or end>len(indices) or covered[start:end].any():
            raise ValueError('Invalid draw skin range')
        palette=np.frombuffer(data,dtype=np.uint8,count=n,offset=off+12).astype(int)
        if np.any(palette>=bone_count):raise ValueError('Draw skin palette exceeds skeleton')
        source,reverse=np.unique(np.asarray(indices[start:end]),return_inverse=True)
        local=ids[source].copy();active=weights[source]>0
        if np.any(local[active]>=n):raise ValueError('Vertex influence exceeds draw palette')
        local[~active]=0;mapped=palette[local];mapped[~active]=0
        out_indices[start:end]=reverse+len(vertices)
        vertices.extend(source.tolist());out_ids.extend(mapped);out_weights.extend(weights[source])
        covered[start:end]=True
    if not covered.all():raise ValueError('Draw skin ranges do not cover triangles')
    # Only referenced vertices belong to the draw. Shared source buffers may
    # also contain body geometry, or reuse a vertex under another palette.
    return out_indices.tolist(),positions[vertices],normals[vertices],uvs[vertices],np.asarray(out_ids),np.asarray(out_weights)


def validate_viewhands(model):
    bones=model['bones']
    def descendants(name):
        found={i for i,b in enumerate(bones) if b['name']==name}
        for i,b in enumerate(bones):
            if b['parent'] in found:found.add(i)
        return found
    coverage={}
    for side in ('L','R'):
        chain=descendants(side+' UpperArm');total=0
        for mesh in model['meshes']:
            used=np.unique(mesh['indices']);ids=np.asarray(mesh['boneIndices'])[used];weights=np.asarray(mesh['weights'])[used]
            total+=int(np.any(np.isin(ids,list(chain)) & (weights>0),axis=1).sum())
        if not total:raise ValueError('Viewhands have no vertices weighted to '+side+' arm')
        coverage[side]=total
    model['armVertexCoverage']=coverage


def read_model(data, name, assets, root, cache, arms=False, player=False):
    bl=blocks(data); refs=extrefs(data)
    bma=next((d for k,d in bl.values() if k=='i3BoneMatrixListAttr'),None)
    selectedLod=None
    if arms or player:
        lods=[d for k,d in bl.values() if k=='i3LOD']
        if len(lods) <= (1 if player else 0):raise ValueError('No corresponding character LOD is present')
        selectedLod=lods[1 if player else 0]
        skeleton=bl[u(selectedLod,8)][1];bma=bl[u(skeleton,8)][1]
    if bma is None: raise ValueError('This model has no supported skeleton')
    count=u(bma,4); bones=[]; worlds=[]
    for i in range(count):
        o=48+i*128; bn=span(bma,o,32).split(b'\0')[0].decode('cp1252'); parent=struct.unpack_from('<i',bma,o+32)[0]
        if parent>=i or parent < -1: raise ValueError('Invalid bone hierarchy')
        m=np.array(struct.unpack_from('<16f',bma,o+48)).reshape(4,4).T
        world=(worlds[parent]@m) if parent>=0 else m; worlds.append(world)
        bones.append(dict(name=bn,parent=parent,local=trs(m),world=trs(world),inverse=np.linalg.inv(world).T.flatten().tolist()))
    children={}; parentnodes={}
    for id,(kind,d) in bl.items():
        if kind in ('i3BoneRef','i3AttrSet','i3Geometry','i3Node','i3Transform'):
            _,p=short(d)
            if d[p:p+4]!=b'INF2': continue
            nc=struct.unpack_from('<Q',d,p+8)[0]
            if nc>100000: raise ValueError('Invalid child count')
            children[id]=list(struct.unpack('<'+'I'*nc,span(d,p+16,nc*4)))
            for child in children[id]: parentnodes[child]=id
    def ancestry(id):
        seen=set()
        while id in bl and id not in seen:
            seen.add(id);yield id
            if id not in parentnodes:break
            id=parentnodes[id]
    allowed=None
    if any(k=='i3LOD' for k,d in bl.values()):
        bmaid=next(i for i,(k,d) in bl.items() if k=='i3BoneMatrixListAttr' and d==bma)
        skids={i for i,(k,d) in bl.items() if k=='i3Skeleton' and u(d,8)==bmaid}
        lod=selectedLod if selectedLod is not None else next(d for k,d in bl.values() if k=='i3LOD' and u(d,8) in skids)
        allowed=set();pending=[u(lod,16)]
        while pending:
            node=pending.pop()
            if node in allowed:continue
            allowed.add(node);pending.extend(children.get(node,[]))
    sockets={}
    if arms:
        for node,(kind,d) in bl.items():
            if kind!='i3Transform':continue
            sn,sp=short(d)
            if not sn.startswith('WeaponPointDummy'):continue
            parent=parentnodes.get(node)
            if parent is None or bl[parent][0]!='i3BoneRef':continue
            bd=bl[parent][1];_,bp=short(bd);nc=struct.unpack_from('<Q',bd,bp+8)[0]
            bi=u(bd,bp+16+nc*4+4)
            master=next((b for k,b in bl.values() if k=='i3BoneMatrixListAttr' and u(b,4)>bi),None)
            if master is None:continue
            boneName=master[48+bi*128:80+bi*128].split(b'\0')[0].decode('cp1252')
            pi=next((i for i,b in enumerate(bones) if b['name']==boneName),None)
            if pi is None:continue
            mid=u(d,len(d)-8);raw=bl[mid][1]
            if len(raw)!=64:raise ValueError('Unsupported socket matrix')
            mat=np.array(struct.unpack('<16f',raw)).reshape(4,4).T
            world=worlds[pi]@mat
            sockets[sn]=len(bones);bones.append(dict(name=sn,parent=pi,local=trs(mat),world=trs(world),inverse=np.linalg.inv(world).T.flatten().tolist()));worlds.append(world)
    slots={'i3TextureBindAttr' :'albedo','i3NormalMapBindAttr':'normal','i3SpecularMapBindAttr':'specular','i3EmissiveMapBindAttr':'emissive'}
    meshes={}; materials={}; warnings=[]
    for id,(kind,d) in bl.items():
        if kind!='i3Geometry' or (allowed is not None and id not in allowed):continue
        mn,_=short(d)
        if not mn:mn='Mesh_'+str(id)
        if player:mn=mn+'_'+str(id)
        if arms and mn.lower()!='model_arms':mn=(mn or 'Viewarms')+'_'+str(id)
        ga=bl[u(d,len(d)-4)][1]; off=4 if ga[:4]==b'GEO2' else 0
        if ga[off]!=4: raise ValueError('Only triangle lists are supported')
        vd=bl[u(ga,off+9)][1]; ix=bl[u(ga,off+13)][1] if u(ga,off+13) else None
        if vd[:4]!=b'VA30':raise ValueError('Unsupported vertex buffer format')
        nv=u(vd,8); stride,rem=divmod(len(vd)-40,nv)
        flag=u(vd,4); uvOffset=12+(12 if flag&2 else 0)+(4 if flag&8 else 0); skinOffset=uvOffset+8
        if rem or stride<skinOffset:raise ValueError('Invalid vertex layout: '+str(stride))
        # Geometry names are labels, not identities. Material sections can share
        # a name and vertex buffer while referencing different triangle lists.
        # LOD selection above already excludes lower-detail geometry.
        if mn in meshes:
            baseName=mn;mn=baseName+'__part_'+str(id)
            while mn in meshes:mn+='_'

        if ix is None:ni=nv;iw=2;ih=0
        elif ix[:4]==b'IIA2': ni=struct.unpack_from('<Q',ix,4)[0]; iw=4 if u(ix,12)==1 else 2; ih=32
        else: ni=struct.unpack_from('<Q',ix,0)[0];iw=2;ih=8
        indices=list(range(nv)) if ix is None else list(struct.unpack('<'+('I' if iw==4 else 'H')*ni,span(ix,ih,ni*iw)))
        if ni%3 or max(indices,default=0)>=nv:raise ValueError('Invalid triangle indices')
        owner=0; mat={}
        for ancestor in ancestry(id):
            k,b=bl[ancestor]
            if k=='i3BoneRef':
                bn,_=short(b);owner=next((i for i,x in enumerate(bones) if x['name']==bn),0);break
        for ancestor in reversed(list(ancestry(id))):
            k,b=bl[ancestor]
            if k=='i3AttrSet':
                _,renderIds=_parse_attr_set(b)
                for ai in renderIds:
                    ak,ab=bl[ai]
                    if ak in slots and len(ab)>=2:
                        ref=refs.get(struct.unpack_from('<H',ab)[0]); texdata=None
                        if not ref:
                            candidates=[n for n in assets if n.endswith('.i3i') and texture_role(n)==slots[ak]]
                            if len(candidates)==1:ref=candidates[0]
                            else:warnings.append('Unresolved '+slots[ak]+' texture for '+mn)
                        if ref:
                            texdata=assets.get(ref.replace('\\','/').lower())
                            if texdata is None:texdata=assets.get(Path(ref).name.lower())
                            if texdata is None:texdata=shared_asset(root,ref,cache)
                            if texdata is None:warnings.append('Missing texture: '+ref)
                        if texdata==b'':mat.pop(slots[ak],None)
                        if texdata:
                            texname=clean(Path(ref).stem)+'_'+hashlib.sha1(texdata).hexdigest()[:8]+'.png'; path=cache/texname
                            try:
                                if not path.exists():texture(texdata,path)
                                mat[slots[ak]]=texname
                            except Exception as ex:warnings.append(str(ex)+': '+ref)
        mk=hashlib.sha1(json.dumps(mat,sort_keys=True).encode()).hexdigest()[:10];materials[mk]=mat
        pos=np.ndarray((nv,3),dtype='<f4',buffer=vd,offset=40,strides=(stride,4)).astype(float)
        norm=np.ndarray((nv,3),dtype='<f4',buffer=vd,offset=52,strides=(stride,4)).astype(float) if flag&2 else np.zeros((nv,3))
        uv=np.ndarray((nv,2),dtype='<f4',buffer=vd,offset=40+uvOffset,strides=(stride,4)).astype(float);uv[:,1]=1-uv[:,1]
        if not all(np.isfinite(v).all() for v in (pos,norm,uv)):raise ValueError('Non-finite vertex')
        influences=(flag>>14)&15; explicit=(flag>>18)&15
        blended=1<=explicit<=3 and influences==explicit+1
        draw_skinned=ga[(25 if off else 21):(29 if off else 25)]==b'SGA1'
        skinned=blended or (draw_skinned and influences==1 and explicit==0)
        if skinned:
            if stride<skinOffset+4+explicit*4:raise ValueError('Truncated skin vertex')
            if not player and not draw_skinned:
                palette=next((b for k,b in bl.values() if k=='i3MemoryBuffer' and len(b)==4+count*4),None)
                if palette is None:raise ValueError('Missing bone palette')
                paletteIds=list(struct.unpack('<'+'I'*count,palette[4:]))
                master=next((b for k,b in bl.values() if k=='i3BoneMatrixListAttr' and u(b,4)>max(paletteIds) and all(b[48+j*128:80+j*128].split(b'\0')[0].decode('cp1252')==bones[i]['name'] for i,j in enumerate(paletteIds))),None)
                if master is None or any(master[48+j*128:80+j*128].split(b'\0')[0].decode('cp1252')!=bones[i]['name'] for i,j in enumerate(paletteIds)):raise ValueError('Unverified bone palette')
            ids=np.ndarray((nv,influences),dtype='u1',buffer=vd,offset=40+skinOffset,strides=(stride,1)).astype(int)
            w=np.ndarray((nv,explicit),dtype='<f4',buffer=vd,offset=44+skinOffset,strides=(stride,4)).astype(float);weights=np.c_[w,1-w.sum(axis=1)]
            weights[np.abs(weights)<1e-6]=0
            ids[weights==0]=0
            if ids.max()>=count or not np.isfinite(weights).all() or weights.min()<0 or np.max(abs(weights.sum(axis=1)-1))>.005:raise ValueError('Invalid skin weights')
        else:
            ids=np.full((nv,1),owner);weights=np.ones((nv,1))
            pos=(worlds[owner]@np.c_[pos,np.ones(nv)].T).T[:,:3];norm=(worlds[owner][:3,:3]@norm.T).T
        if skinned:
            indices,pos,norm,uv,ids,weights=remap_draw_skin(ga,bl,indices,pos,norm,uv,ids,weights,count)
        meshes[mn]=dict(name=mn,positions=pos.tolist(),normals=norm.tolist(),uvs=uv.tolist(),indices=indices,boneIndices=ids.tolist(),weights=weights.tolist(),material=mk)
    if not meshes:raise ValueError('No supported meshes found')
    model=dict(name=name,bones=bones,meshes=list(meshes.values()),materials=materials,warnings=sorted(set(warnings)),sockets=sockets)
    if arms:validate_viewhands(model)
    return model

def decode(path, root, cache):
    data=path.read_bytes()
    key=hashlib.sha1(json.dumps([str(root),VERSION,DECODER_REV]).encode()+data).hexdigest()
    out=INDEX_HOME/'animations'/(key+'.animation.json');out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():
        result=decode_pack(data,str(path))
        save_json(out,result)
    return json.loads(out.read_text())['clips']

def normalized(s):return re.sub('[^a-z0-9]','',s.lower())
METADATA_CACHE={}
def weapon_records(root):
    p=Path(root)/'Pack/Script.i3Pack';key=(str(p),p.stat().st_size,p.stat().st_mtime_ns)
    if key not in METADATA_CACHE:METADATA_CACHE[key]=persistent_index('metadata-ui-v03',key,lambda:load_weapon_metadata_index(root)['records'])
    return METADATA_CACHE[key]
def resource_identity(v):return normalized(re.sub(r'^weapon[_-]*','',Path(str(v).replace('\\','/')).stem,flags=re.I))
def is_dual_weapon(v):return 'dual' in resource_identity(v).replace('dualmagazine','').replace('dualmag','')
def weapon_record(root,source):
    key=resource_identity(source.stem);ranked=[]
    for r in weapon_records(root):
        score=max((score for field,score in [('key',3),('_ResName_I3S',2),('_ResName',1)] if r.get(field) and resource_identity(r[field])==key),default=0)
        if score:ranked.append((score,r))
    if not ranked:
        try:
            names={resource_identity(n) for n in package_assets(source) if n.endswith('.i3s')}
            for r in weapon_records(root):
                score=max((score for field,score in [('key',5),('_ResName_I3S',4),('_ResName',3)] if r.get(field) and resource_identity(r[field]) in names),default=0)
                if score:ranked.append((score,r))
        except (ValueError,OSError):pass
    return max(ranked,key=lambda x:(x[0],is_dual_weapon(x[1].get('key',''))==is_dual_weapon(source.stem),not x[1].get('key','').lower().endswith('dummy'),bool(x[1].get('ClassMeta'))))[1] if ranked else {}

CATEGORY_IDS={103:'Assault rifles',104:'SMGs',105:'Sniper rifles',106:'Shotguns',110:'Machine guns',116:'Launchers',117:'Machine guns',118:'SMGs',119:'Dinosaur',135:'Shotguns',202:'Pistols',213:'Pistols',214:'Pistols',230:'Shotguns',234:'Bows',301:'Knives / melee',315:'Knives / melee',323:'Knives / melee',407:'Grenades',411:'Grenades',412:'Equipment',508:'Grenades',527:'Grenades',528:'Equipment',5009:'Equipment'}
CATALOG_CACHE={}
def build_family_catalog(root):
    root=Path(root); paths=sorted((p for p in (root/'Pack').glob('Weapon*.i3Pack') if p.stem.lower().startswith(('weapon_','weapon-'))),key=lambda p:p.name.lower())
    stamp=(str(root),tuple((p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in paths))
    if stamp in CATALOG_CACHE:return CATALOG_CACHE[stamp]
    records=weapon_records(root)
    def identity(v):return normalized(re.sub(r'^weapon[_-]*','',Path(str(v).replace('\\','/')).stem,flags=re.I))
    def dual(v):return is_dual_weapon(v)
    aliases={}; roots={}
    for r in records:
        for field in ('key','_ResName_I3S','_ResName'):
            v=r.get(field)
            if v:aliases.setdefault(identity(v),[]).append(r)
        v=r.get('_ResName')
        if v:roots.setdefault(identity(v),v)
        linked=r.get('LinkedToCharaAI')
        if linked:roots.setdefault(identity(linked),linked)
    installed={identity(p.stem):p.name for p in paths}
    roots.update({k:Path(v).stem[7:] for k,v in installed.items()})
    # Learn spelling aliases from authored model/provider pairs, retaining the
    # shared variant suffix (e.g. Kukri_Comic -> Kukrii + Comic).
    from difflib import SequenceMatcher
    prefix_votes={}
    for r in records:
        model=r.get('_ResName_I3S');provider=r.get('_ResName')
        if not model or not provider:continue
        mt=re.split('[_ -]+',str(model));kt=re.split('[_ -]+',str(r.get('key','')))
        while len(mt)>1 and len(kt)>1 and identity(mt[-1])==identity(kt[-1]):mt.pop();kt.pop()
        a=identity('_'.join(mt));b=identity(provider)
        if len(a)>=4 and SequenceMatcher(None,a,b).ratio()>=0.65 and a not in roots and a!=identity(model) and b in roots and dual(a)==dual(b):prefix_votes.setdefault(a,set()).add(b)
    prefix_alias={a:next(iter(v)) for a,v in prefix_votes.items() if len(v)==1}
    groups={}
    for p in paths:
        key=identity(p.stem)
        # Only whole separator-delimited prefixes can own a cosmetic variant.
        tokens=re.split('[_ -]+',p.stem[7:])
        options=[identity('_'.join(tokens[:i])) for i in range(1,len(tokens)+1)]
        parent=next((prefix_alias.get(v,v) for v in options if (v in roots or v in prefix_alias) and dual(v)==dual(key)),key)
        groups.setdefault(parent,[]).append(p.name)
    result=[]
    for key,vs in groups.items():
        base=installed.get(key); missing=base is None
        if base is None:
            base=vs[0]
            def base_priority(v):
                rr=aliases.get(identity(Path(v).stem),[])
                return (min((int(r.get('ITEMID') or 99999999) for r in rr),default=99999999),v.lower())
            for v in sorted(vs,key=base_priority):
                try:
                    if any(n.lower().endswith('.i3s') for n,d in pack((root/'Pack'/v).read_bytes())):base=v;break
                except (ValueError,struct.error):continue
        matches=aliases.get(key,[])
        if not matches:
            matches=[r for v in vs for r in aliases.get(identity(Path(v).stem),[])]
        categories={CATEGORY_IDS.get(int(r.get('ITEMID') or 0)//1000,'Other / unclassified') for r in matches}
        category=next(iter(categories)) if len(categories)==1 else 'Other / unclassified'
        if category=='Other / unclassified':
            # Custom packs can preserve an authored resource/animation family.
            try:
                aa=package_assets(root/'Pack'/base);rr=[]
                for n,d in aa.items():
                    if n.endswith('.i3chr'):rr+=list(extrefs(d).values())+ [x.decode('cp1252') for x in re.findall(rb'Weapon[/\\][A-Za-z0-9_./\\-]+',d)]
                hints={identity(x.replace('\\','/').split('/')[1]) for x in rr if x.lower().startswith('weapon') and len(x.replace('\\','/').split('/'))>1}
                cats={CATEGORY_IDS.get(int(r.get('ITEMID') or 0)//1000,'Other / unclassified') for h in hints for r in aliases.get(h,[])}
                if len(cats)==1:category=next(iter(cats))
            except (ValueError,struct.error):pass
        if category=='Other / unclassified':
            rr=weapon_record(root,root/'Pack'/base)
            category=CATEGORY_IDS.get(int(rr.get('ITEMID') or 0)//1000,category)
        if category=='Other / unclassified':
            names={identity(Path(base).stem)}|{identity(n) for n in package_assets(root/'Pack'/base) if n.endswith('.i3s')}
            hints=set()
            for ui in (root/'Pack').glob('UI_Weapon*.i3Pack'):
                m=re.match(r'UI_Weapon(Assault|Sniper|Knife|Handgun|SMG|Shotgun|MachineGun)_?(.*)',ui.stem,re.I)
                if m and any(identity(m[2])==n or (len(identity(m[2]))>=4 and n.startswith(identity(m[2]))) for n in names):hints.add({'assault':'Assault rifles','sniper':'Sniper rifles','knife':'Knives / melee','handgun':'Pistols','smg':'SMGs','shotgun':'Shotguns','machinegun':'Machine guns'}[m[1].lower()])
            if len(hints)==1:category=hints.pop()
        result.append(dict(base=base,name=roots.get(key,Path(base).stem[7:]),variants=[base]+[v for v in vs if v!=base],baseUnavailable=missing,category=category))
    virtuals={}
    for r in records:
        if r.get('ClassMeta')!='WeaponDualKnife':continue
        provider=installed.get(identity(r.get('_ResName','')))
        if not provider:continue
        baseRecord=weapon_record(root,root/'Pack'/provider)
        if int(baseRecord.get('ITEMID') or 0)//1000!=301:continue
        keyname=r.get('key','')
        if keyname.endswith(('_D','_EV')):continue
        virtuals.setdefault(r['_ResName'],[]).append('Weapon_'+keyname+'.virtual')
    for name,variants in virtuals.items():
        result.append(dict(base=variants[0],name=name+' Dual',variants=variants,baseUnavailable=False,category='Knives / melee'))
    # One cosmetic model can serve both a single weapon and its authored dual
    # replacement. Keep the base file in the single family; expose record-backed
    # dual variants so loading retains the correct class and animation identity.
    for r in records:
        if r.get('ClassMeta')!='WeaponDualSMG' or not r.get('_ResName_I3S'):continue
        if identity(r['_ResName_I3S']) not in installed:continue
        bases=[b for b in records if b.get('ClassMeta')==r['ClassMeta'] and b.get('_ResName')==r.get('_ResName') and not b.get('_ResName_I3S') and identity(b.get('key','')) in installed]
        if not bases:continue
        basefile=installed[identity(bases[0]['key'])]
        group=next((g for g in result if g['base']==basefile),None)
        if group is not None:
            virtual='Weapon_'+r['key']+'.virtual'
            if virtual not in group['variants']:group['variants'].append(virtual)
    result.sort(key=lambda g:(g['category'],g['name'].lower()))
    CATALOG_CACHE.clear();CATALOG_CACHE[stamp]=result
    return result

def family_catalog(root):
    root=Path(root);items=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted((root/'Pack').glob('*.i3Pack'))]
    return persistent_index('families-v01b',(str(root),items),lambda:build_family_catalog(root))

def build_viewhands_catalog(root):
    result=[]
    for p in sorted((Path(root)/'Pack').glob('Chara*.i3Pack')):
        try:
            for name,d in pack(p.read_bytes()):
                if not name.lower().endswith('.i3s'):continue
                bl=blocks(d);lod=next((b for k,b in bl.values() if k=='i3LOD'),None)
                if lod is None:continue
                bma=bl[u(bl[u(lod,8)][1],8)][1]
                names=[bma[48+i*128:80+i*128].split(b'\0')[0].decode('cp1252') for i in range(u(bma,4))]
                if not {'R Hand','L Hand'}<=set(names):continue
                pending=[u(lod,16)];seen=set();geometries=0
                while pending:
                    node=pending.pop()
                    if node in seen or node not in bl:continue
                    seen.add(node);kind,b=bl[node]
                    if kind=='i3Geometry':geometries+=1
                    if kind in ('i3BoneRef','i3AttrSet','i3Geometry','i3Node','i3Transform'):
                        _,off=short(b)
                        if b[off:off+4]==b'INF2':
                            count=struct.unpack_from('<Q',b,off+8)[0];pending.extend(struct.unpack('<'+'I'*count,span(b,off+16,count*4)))
                if geometries:
                    result.append(dict(base=p.name,name=p.stem.removeprefix('Chara_').replace('_',' '),variants=[p.name],category='Viewhands',model=name.lower(),baseUnavailable=False))
                    break
        except (ValueError,struct.error,KeyError):continue
    return result

def viewhands_catalog(root):
    root=Path(root);items=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted((root/'Pack').glob('Chara*.i3Pack'))]
    return persistent_index('hands',(str(root),items),lambda:build_viewhands_catalog(root))

def load_viewhands(req):
    root=Path(req['root']);source=root/'Pack'/req['pack'];cache=Path(req['cache']);cache.mkdir(parents=True,exist_ok=True)
    entry=next((g for g in viewhands_catalog(root) if g['base']==source.name),None)
    if entry is None:raise ValueError('No viewhand model found in this package')
    assets=package_assets(source);d=assets[entry['model']];name=source.stem.removeprefix('Chara_')
    arms=read_character(d,name+'_viewhands',assets,root,cache,arms=True)
    player=None;warnings=[]
    try:player=read_character(d,name+'_playermodel',assets,root,cache,player=True)
    except (ValueError,StopIteration,KeyError) as e:warnings.append('Corresponding player model unavailable: '+str(e))
    scene=dict(version=VERSION,kind='viewhands',source=str(source),baseSource=str(source),modelChoices=[],weapon=None,arms=arms,player=player,clips=[],cache=str(cache.resolve()),fps=30,warnings=warnings)
    save_json(cache/'scene.json',scene)
    return dict(ok=True,message='Viewhands loaded. Player model exports separately.',scene=str((cache/'scene.json').resolve()))

def preload_category(req):
    cache=Path(req['cache']);cache.mkdir(parents=True,exist_ok=True);loaded={};errors={}
    for i,asset in enumerate(req['packs']):
        child={**req,'pack':asset,'cache':str(cache/clean(asset))}
        try:loaded[asset]=load(child)['scene']
        except Exception as e:errors[asset]=str(e)
        save_json(cache/'progress.json',dict(done=i+1,total=len(req['packs']),loaded=loaded,errors=errors))
    return dict(ok=True,message=f"Preloaded {len(loaded)} assets; {len(errors)} unavailable.",loaded=loaded,details='\n'.join(k+': '+v for k,v in errors.items()))

PACKAGE_CACHE={}
def package_assets(path):
    path=Path(path);key=(str(path),path.stat().st_size,path.stat().st_mtime_ns)
    if key not in PACKAGE_CACHE:
        if len(PACKAGE_CACHE)>16:PACKAGE_CACHE.clear()
        PACKAGE_CACHE[key]={Path(n).name.lower():d for n,d in pack(path.read_bytes())}
    return PACKAGE_CACHE[key]

def gallery(req):
    from studio_i3 import i3i_to_dds
    root=Path(req['root']);cache=Path(req['cache']);cache.mkdir(parents=True,exist_ok=True)
    paths=sorted((root/'Pack').glob('UI*Weapon*.i3Pack'))
    def build():
        index={}
        for p in paths:
            try:
                entries=list(pack(p.read_bytes()))
                if any(n.lower().endswith('.i3s') for n,d in entries):continue
                for n,d in entries:
                    if n.lower().endswith('.i3i') and d[:4]==b'I3IB' and struct.unpack_from('<HH',d,6)==(512,512):
                        index.setdefault(resource_identity(n),[p.name,n])
            except (ValueError,struct.error):continue
        return index
    index=persistent_index('menu-icons-v02b',[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths],build)
    records={resource_identity(r['key']):r for r in weapon_records(root)};items=[]
    aliases={}
    for r in records.values():
        for field in ('_ResName_I3S','_ResName'):
            if r.get(field):aliases.setdefault(resource_identity(r[field]),[]).append(r)
    familyByAsset={v:g for g in family_catalog(root) for v in g['variants']}
    for asset in req['assets']:
        key=Path(asset).stem.removeprefix('Weapon_');identity=resource_identity(key);record=records.get(identity,{})
        linked=aliases.get(identity,[])
        if not record and linked:record=next((r for r in linked if not r.get('_ResName_I3S')),linked[0])
        candidates=[record.get('UiPath',''),key,record.get('_ResName_I3S','')]
        family=familyByAsset.get(asset,{})
        familykey=resource_identity(family.get('base',''));base=records.get(familykey,{})
        if not base:
            linkedBase=aliases.get(familykey,[]);base=next((r for r in linkedBase if not r.get('_ResName_I3S')),linkedBase[0] if linkedBase else {})
        candidates += [base.get('UiPath','')]
        match=next((index[resource_identity(k)] for k in candidates if k and resource_identity(k) in index),None)
        item=dict(asset=asset,name=key.replace('_',' '),icon='',note='Unresolved menu reference')
        # Follow authored shape indices into the equipment atlas (88 used cells per page; five columns).
        shape=record.get('_UIShapeIndex') if record else base.get('_UIShapeIndex');atlas=root/'Pack/UI_WeaponShape.i3Pack';crop=None
        if shape is not None and int(shape)>=0:
            page,cell=divmod(int(shape),88);atlasName='weapon_select'+str(page)+'.i3i'
            if atlas.is_file() and atlasName in package_assets(atlas):
                match=[atlas.name,atlasName];col,row=cell%5,cell//5;crop=(col*204+4,row*57+1,col*204+204,row*57+56)
        if not match and record.get('UiPath'):
            shared=next((r for r in records.values() if r.get('UiPath')==record.get('UiPath') and not r.get('_ResName_I3S') and r.get('_UIShapeIndex') is not None),{})
            sharedIndex=shared.get('_UIShapeIndex')
            if sharedIndex is not None:
                page,cell=divmod(int(sharedIndex),88);atlasName='weapon_select'+str(page)+'.i3i'
                if atlas.is_file() and atlasName in package_assets(atlas):
                    match=[atlas.name,atlasName];shape=sharedIndex;col,row=cell%5,cell//5;crop=(col*204+4,row*57+1,col*204+204,row*57+56)
        if not match and not asset.endswith('.virtual'):
            try:
                resolved=weapon_record(root,root/'Pack'/asset)
                ui=resolved.get('UiPath','');match=index.get(resource_identity(ui)) if ui else None
                if match:item['note']='Resolved from internal model record'
            except (ValueError,OSError,struct.error):pass
        if match:
            p=root/'Pack'/match[0];stamp=hashlib.sha256(('menu-crop-v03'+str(crop)+str(p)+str(p.stat().st_mtime_ns)+match[1]).encode()).hexdigest();out=cache/(stamp+'.rgba')
            try:
                if not out.exists():
                    data=package_assets(p)[Path(match[1]).name.lower()];im=Image.open(io.BytesIO(i3i_to_dds(data))).convert('RGBA');im=im.crop(crop or (0,0,208,56));im.thumbnail((192,64));canvas=Image.new('RGBA',(192,160));canvas.paste(im,((192-im.width)//2,(160-im.height)//2));im=canvas;out.write_bytes(struct.pack('<II',*im.size)+im.tobytes())
                item.update(icon=str(out.resolve()),note=match[0]+' / '+match[1]+(' / shape '+str(shape) if crop else ' / UiPath'))
            except Exception as e:item['note']='Menu reference decode failed: '+str(e)
        items.append(item)
    return dict(ok=True,message='Variant gallery ready.',gallery=items)

def texture_role(name):
    n=normalized(Path(name).stem)
    if any(x in n for x in ('normal','norm','nomal')):return 'normal'
    if 'spec' in n:return 'specular'
    if 'emiss' in n:return 'emissive'
    if 'ref' in n:return 'reflection'
    return 'albedo'

def load(req):
    root=Path(req['root']);cache=Path(req['cache']);path=cache/'scene.json'
    families={'SWAT_'+req.get('arms','Male')}
    if req['pack'].lower().startswith('chara'):families.update(p.name for p in (root/'Chara').iterdir() if p.is_dir() and p.name.lower() in req['pack'].lower())
    files=list((root/'Pack').glob('*.i3Pack'))+[p for family in families for p in (root/'Chara'/family).glob('*.i3AnimPack')]
    stamp=hashlib.sha256(json.dumps([VERSION,PIPELINE_REV,str(root),{k:v for k,v in req.items() if k in ('pack','model','arms','character')},[(str(p.relative_to(root)),st.st_size,st.st_mtime_ns) for p in sorted(files) for st in [p.stat()]]]).encode()).hexdigest()
    if path.exists():
        try:
            saved=json.loads(path.read_text())
            if saved.get('inputStamp')==stamp and saved.get('samplingRevision')==DECODER_REV:return dict(ok=True,scene=str(path.resolve()),message='Loaded and previewed cached assets.')
        except (ValueError,OSError):pass
    actual=dict(req)
    if req['pack'].endswith('.virtual'):
        recordKey=req['pack'][7:-8]
        record=next(r for r in weapon_records(root) if r.get('key')==recordKey)
        provider=next(p for p in (root/'Pack').glob('Weapon*.i3Pack') if resource_identity(p.stem)==resource_identity(record['_ResName']))
        variant=next((p for p in (root/'Pack').glob('Weapon*.i3Pack') if record.get('_ResName_I3S') and resource_identity(p.stem)==resource_identity(record['_ResName_I3S'])),provider)
        actual.update(pack=variant.name,recordKey=recordKey)
    actual['_deferScene']=True
    result=load_uncached(actual)
    scene=result.pop('_scene',None)
    if scene is None:scene=json.loads(Path(result['scene']).read_text())
    scene['pipelineRevision']=PIPELINE_REV;scene['samplingRevision']=DECODER_REV;scene['inputStamp']=stamp;scene['assetId']=req['pack'];scene['assetName']=actual.get('recordKey',Path(req['pack']).stem.removeprefix('Weapon_'));save_json(path,scene)
    return result

def texture_remaps(root):
    from studio_metadata import _decode_pef,_global_block_id,_parse_key,_dotnet_string
    script=Path(root)/'Pack/Script.i3Pack'
    def build():
        data=next(d for n,d in pack(script.read_bytes()) if n.lower()=='texture_change_weapon.pef')
        parsed=_decode_pef(data,'texture_change_Weapon.Pef');nodes={_global_block_id(b):b for b in parsed.blocks};result={}
        for block in parsed.blocks:
            if block.type_name!='i3RegKey':continue
            name,children,_=_parse_key(block.data)
            for cid in children:
                child=nodes.get(cid)
                if not child or child.type_name!='i3RegKey':continue
                lod,_,values=_parse_key(child.data)
                if lod.lower()!='lod 0':continue
                mappings={}
                for vid in values:
                    value=nodes.get(vid)
                    if not value or value.type_name!='i3RegString':continue
                    src,pos=_dotnet_string(value.data,0);pos+=4
                    marker=value.data[pos:pos+4];size=u(value.data,pos+4);width=2 if marker==b'RGS3' else 1
                    if marker not in (b'RGS2',b'RGS3'):raise ValueError('Unsupported texture-remap string version')
                    dst=span(value.data,pos+8,size*width).decode('utf-16le' if width==2 else 'cp1252').rstrip('\0')
                    if src and dst:mappings[src.replace('\\','/').lower()]=dst.replace('\\','/')
                if mappings:result[resource_identity(name)]=mappings
        return result
    return persistent_index('texture-remaps-v04',(str(script),script.stat().st_size,script.stat().st_mtime_ns),build)

def apply_texture_remaps(root,source,record,assets,selectedAssets,cache):
    table=texture_remaps(root);keys=[source.stem,record.get('_ResName_I3S',''),record.get('key','')]
    mapping=next((table[resource_identity(k)] for k in keys if k and resource_identity(k) in table),{})
    applied={};byName={}
    for src,dst in mapping.items():
        if src.lower()==dst.lower():continue
        if dst.lower()=='(null)':
            assets[src]=b'';byName.setdefault(Path(src).name.lower(),[]).append(b'');applied[src]=dst;continue
        compiled=str(Path(dst).with_suffix('.i3i')) if Path(dst).suffix.lower() in ('.tga','.dds') else dst
        name=Path(compiled).name.lower();data=selectedAssets.get(name) or assets.get(name)
        if data is None:data=shared_asset(root,compiled,cache)
        if data is None:raise ValueError('Variant texture referenced by the game is missing: '+dst)
        assets[src]=data;byName.setdefault(Path(src).name.lower(),[]).append(data);applied[src]=dst
    for name,values in byName.items():
        if all(v==values[0] for v in values):assets[name]=values[0]
    return applied

def load_uncached(req):
    if req['pack'].lower().startswith('chara'):return load_viewhands(req)
    root=Path(req['root']); source=root/'Pack'/req['pack']; cache=Path(req['cache']);cache.mkdir(parents=True,exist_ok=True)
    if not source.is_file() or source.parent.resolve()!=(root/'Pack').resolve():raise ValueError('Choose a package from this client Pack folder')
    selectedAssets=package_assets(source)
    family=next((g for g in family_catalog(root) if source.name in g['variants']),None)
    record=next((r for r in weapon_records(root) if r.get('key')==req.get('recordKey')),None) or weapon_record(root,source)
    provider=next((p for p in (root/'Pack').glob('Weapon*.i3Pack') if record.get('_ResName') and resource_identity(p.stem)==resource_identity(record['_ResName'])),None)
    baseSource=provider or root/'Pack'/(family['base'] if family else source.name)
    if not record and baseSource!=source:record=weapon_record(root,baseSource)
    baseAssets=package_assets(baseSource) if baseSource!=source else selectedAssets
    assets={**baseAssets,**selectedAssets}
    selectedModels=[(n,d) for n,d in selectedAssets.items() if n.endswith('.i3s')]
    models=selectedModels or [(n,d) for n,d in baseAssets.items() if n.endswith('.i3s')]
    if not models:raise ValueError('No geometry is present in this family base package. This is a resource-only package: '+source.name)
    appliedRemaps=apply_texture_remaps(root,source,record,assets,selectedAssets,cache)
    if not selectedModels and source!=baseSource and not appliedRemaps:
        # Skin-only packages override a unique texture role. Ambiguous roles are
        # rejected rather than silently applying a texture to the wrong part.
        def component(n,identities):
            tokens=re.findall('[a-z0-9]+',Path(n).stem.lower())
            if tokens and tokens[-1] in ('diff','diffuse','albedo','normal','norm','nomal','spec','specular','emissive','emiss','ref','refmask'):tokens=tokens[:-1]
            prefixes={resource_identity(v) for v in identities if v}
            cuts=[i for i in range(1,len(tokens)+1) if ''.join(tokens[:i]) in prefixes]
            return ''.join(tokens[max(cuts):]) if cuts else None
        selectedIdentities=[source.stem,record.get('_ResName_I3S'),family['name'] if family else '']
        baseIdentities=[baseSource.stem,record.get('_ResName'),family['name'] if family else '']
        overrides={}
        for n,d in selectedAssets.items():
            if n.endswith('.i3i'):
                part=component(n,selectedIdentities)
                if part is not None:overrides.setdefault((texture_role(n),part),[]).append(d)
        for n,d in baseAssets.items():
            if n.endswith('.i3i'):
                part=component(n,baseIdentities);replacement=overrides.get((texture_role(n),part),[])
                if part is not None and len(replacement)==1:assets[n]=replacement[0]
    modelname=req.get('model','');desired=resource_identity(record.get('_ResName_I3S') or record.get('_ResName') or '');chosen=next((x for x in models if x[0]==modelname),next((x for x in models if desired and resource_identity(x[0])==desired),max(models,key=lambda x:len(x[1]))))
    weapon=read_model(chosen[1],source.stem.removeprefix('Weapon_'),assets,root,cache)
    actions={}
    chrs=[(n,d) for n,d in assets.items() if n.endswith('.i3chr')]
    chrdata=next((d for n,d in chrs if Path(n).stem==Path(chosen[0]).stem),chrs[0][1] if chrs else None)
    if chrdata:actions=weapon_actions(chrdata)
    wclips=[]
    animationAssets={**baseAssets,**selectedAssets}
    for baseRecord in weapon_records(root):
        if baseRecord.get('ClassMeta')!=record.get('ClassMeta') or baseRecord.get('_ResName')!=record.get('_ResName') or baseRecord.get('_ResName_I3S'):continue
        if not (record.get('ClassMeta') or '').startswith('WeaponDual'):continue
        sibling=next((p for p in (root/'Pack').glob('Weapon*.i3Pack') if resource_identity(p.stem)==resource_identity(baseRecord.get('key',''))),None)
        if sibling:animationAssets={**package_assets(sibling),**animationAssets}
    if not any(n.endswith('.i3animpack') for n in animationAssets) and chrdata:
        resources=list(actions.values())+list(extrefs(chrdata).values())
        for resource in resources:
            parts=resource.replace('\\','/').split('/')
            if len(parts)<3 or parts[0].lower()!='weapon':continue
            target=normalized('Weapon_'+parts[1])
            sibling=next((p for p in (root/'Pack').glob('Weapon*.i3Pack') if normalized(p.stem)==target),None)
            if sibling:
                found=package_assets(sibling)
                if any(n.endswith('.i3animpack') for n in found):animationAssets=found;break
    for n,d in animationAssets.items():
        if n.endswith('.i3animpack'):
            ap=cache/clean(n);ap.write_bytes(d);wclips+=decode(ap,root,cache)
    # Resolve explicitly referenced external animation resources as well.
    present={normalized(c['name']) for c in wclips}
    missing={normalized(ref):ref for ref in actions.values() if normalized(ref) not in present}
    for resource in list(missing.values()):
        parts=resource.replace('\\','/').split('/')
        if len(parts)<3 or parts[0].lower()!='weapon':continue
        sibling=next((p for p in (root/'Pack').glob('Weapon*.i3Pack') if resource_identity(p.stem)==resource_identity(parts[1])),None)
        if not sibling:continue
        for n,d in package_assets(sibling).items():
            if not n.endswith('.i3animpack'):continue
            ap=cache/('referenced_'+clean(sibling.stem)+'_'+clean(n));ap.write_bytes(d)
            for c in decode(ap,root,cache):
                if normalized(c['name']) in missing and normalized(c['name']) not in present:wclips.append(c);present.add(normalized(c['name']))
    wclips=list({normalized(c['name']):c for c in wclips}.values())
    arms=None; player=None; aclips=[]; gender=req.get('arms','Male')
    if gender in ('Male','Female'):
        character=req.get('character','SWAT')
        if character not in ('SWAT','REBEL'):raise ValueError('Unsupported character')
        armpack=root/'Pack'/('Chara_'+character+'_'+gender+'.i3Pack')
        armassets={Path(n).name.lower():d for n,d in pack(armpack.read_bytes())}
        arms=read_character(armassets[character.lower()+'_'+gender.lower()+'.i3s'],character+'_'+gender+'_viewarms',armassets,root,cache,True)
        player=read_character(armassets[character.lower()+'_'+gender.lower()+'.i3s'],character+'_'+gender+'_playermodel',armassets,root,cache,player=True)
        candidates=[record.get('LinkedToCharaAI',''),Path(chosen[0]).stem,source.stem.removeprefix('Weapon_'),baseSource.stem.removeprefix('Weapon_')]
        aclips=character_clips(root,gender,record,candidates,cache)
    isDual=is_dual_weapon(source.stem) or ((record.get('ClassMeta') or '').startswith('WeaponDual') and record.get('ClassMeta') not in ('WeaponDualMagazine','WeaponDualCIC'))
    clips=[]
    aclips=list({Path(c['name']).stem.lower():c for c in sorted(aclips,key=lambda c:'/common/' not in c['name'].lower())}.values())
    for c in aclips:
        stem=Path(c['name']).stem
        wc=weapon_companion(stem,wclips,actions,gender,'Right' if isDual else '')
        clips.append(dict(name=stem,source=c['name'],duration=c['duration'],arms=c,weapon=wc))
        if isDual:clips[-1]['leftWeapon']=weapon_companion(stem,wclips,actions,gender,'Left')
    if not isDual:
        for c in wclips:
            leaf=Path(c['name']).stem.lower()
            if '3pv' in leaf or (gender=='Male' and 'female' in leaf) or (gender=='Female' and 'male' in leaf and 'female' not in leaf):continue
            if 'dual' in Path(c['name']).stem.lower():continue
            clips.append(dict(name='Weapon / '+Path(c['name']).stem,source=c['name'],duration=c['duration'],arms=None,weapon=c))
    scene=dict(category=family.get('category') if family else None,textureRemaps=appliedRemaps,version=VERSION,source=str(source),baseSource=str(baseSource),resourceRecord=record,selectedModel=chosen[0],modelChoices=[n for n,d in models],weapon=weapon,arms=arms,player=player,clips=clips,cache=str(cache.resolve()),fps=30)
    resolve_attachments(scene)
    if isDual:assemble_dual(scene)
    result=dict(ok=True,scene=str((cache/'scene.json').resolve()),message=f"Loaded {len(weapon['meshes'])} weapon parts and {len(clips)} animation choices.")
    if req.get('_deferScene'):result['_scene']=scene
    else:save_json(cache/'scene.json',scene)
    return result

def matching_common_folder(entries,matched,dual):
    # Select a shared idle only when the authored pullout endpoint proves the
    # same arm pose. This chooses resources; it does not fit or transform a rig.
    stamp=[(str(ap),ap.stat().st_size,ap.stat().st_mtime_ns) for ap,_ in entries]
    def find():
        packs={}
        def pose(ap,name,end):
            if ap not in packs:packs[ap]=load_i3animpack(ap)
            pack=packs[ap];clip=next(c for c in pack.clips if c.name.replace('\\','/')==name)
            return pack.sample(clip,clip.duration if end else 0,loop=False)
        targets=[(ap,n) for ap,names in entries for n in names if n.rsplit('/',1)[0] in matched and normalized(Path(n).stem) in ('change','changedual')]
        if not targets:return None
        target=pose(*targets[0],True);scores=[]
        for ap,names in entries:
            for n in names:
                if normalized(n.split('/')[-2]) not in ('common','conmmon') or normalized(Path(n).stem)!='attackidle':continue
                if ('dual' in n.lower().split('/1pv/')[1].split('/')[0])!=dual:continue
                candidate=pose(ap,n,False);angles=[]
                for bone in ('R Hand','L Hand','R UpperArm','L UpperArm','R Forearm','L Forearm'):
                    if bone not in target or bone not in candidate:continue
                    a=target[bone].rotation;b=candidate[bone].rotation
                    norm=math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
                    if norm==0:continue
                    dot=abs(sum(x*y for x,y in zip(a,b)))/norm
                    angles.append(math.degrees(2*math.acos(min(1,dot))))
                if len(angles)==6 and max(angles)<0.05:scores.append((sum(angles),n.rsplit('/',1)[0]))
        folders={folder.lower() for _,folder in scores}
        return min(scores)[1] if len(folders)==1 else None
    return persistent_index('common-binding-v01',[stamp,sorted(matched),dual],find)

def character_clips(root,gender,record,candidates,cache):
    candidates={normalized(n) for n in candidates if n}
    entries=[]
    for ap in (root/'Chara'/('SWAT_'+gender)).glob('*.i3AnimPack'):
        d=ap.read_bytes()
        if d[:4] not in (b'APF1',b'APF2',b'APF3'):continue
        names=[span(d,184+i*284,260).split(b'\0')[0].decode('cp1252').replace('\\','/') for i in range(u(d,4))]
        entries.append((ap,[n for n in names if '/1pv/' in n.lower()]))
    dual=(record.get('ClassMeta') or '').startswith('WeaponDual') and record.get('ClassMeta') not in ('WeaponDualMagazine','WeaponDualCIC') or any(is_dual_weapon(n) for n in candidates)
    matched=set()
    for ap,names in entries:
        for n in names:
            folder=n.rsplit('/',1)[0];parts=folder.lower().split('/1pv/')[1].split('/');group=normalized(parts[0]);family=normalized(parts[-1])
            clipDual='dual' in group or 'dual' in normalized(Path(n).stem)
            if clipDual!=dual:continue
            direct=(family in candidates or (dual and family.replace('dual','') in {v.replace('dual','') for v in candidates})) and family not in ('common','conmmon')
            if direct and family in group and group!=family and group.replace(family,'') not in ''.join(candidates):direct=False
            linked=normalized(record.get('LinkedToCharaAI',''))
            classpath=group==linked and (family in ('common','conmmon') or family in candidates)
            if direct or classpath:matched.add(folder)
    if not matched and record.get('ClassMeta') in ('WeaponKnife','WeaponDualKnife'):
        matched={n.rsplit('/',1)[0] for ap,names in entries for n in names if ('/1pv/dualknife/common/' if dual else '/1pv/knife/common/') in n.lower()}
    parents={f.rsplit('/',1)[0] for f in matched}
    defaults={'Sniper rifles':{'snifferrifle','sniperrifle'},'Assault rifles':{'assultrifle','assaultrifle'},'SMGs':{'smg'},'Pistols':{'handgun'},'Knives / melee':{'dualknife' if dual else 'knife'},'Shotguns':{'shotgun'},'Machine guns':{'machinegun'}}
    has_common=any(normalized(n.split('/')[-2]) in ('common','conmmon') and n.rsplit('/',2)[0] in parents for ap,names in entries for n in names)
    if matched and not has_common:
        verified=matching_common_folder(entries,matched,dual)
        if verified:matched.add(verified);parents.add(verified.rsplit('/',1)[0]);has_common=True
    if matched and not has_common:
        groups=defaults.get(CATEGORY_IDS.get(int(record.get('ITEMID') or 0)//1000,''),set())
        for ap,names in entries:
            for n in names:
                if normalized(n.split('/')[-2]) in ('common','conmmon') and normalized(n.lower().split('/1pv/')[1].split('/')[0]) in groups:matched.add(n.rsplit('/',1)[0])
    selected={n for ap,names in entries for n in names if n.rsplit('/',1)[0] in matched or (normalized(n.split('/')[-2]) in ('common','conmmon') and n.rsplit('/',2)[0] in parents)}
    selected={n for n in selected if (('dual' in normalized(n.lower().split('/1pv/')[1].split('/')[0]) or 'dual' in normalized(Path(n).stem))==dual)}
    result=[]
    for ap,names in entries:
        if any(n in selected for n in names):result.extend(c for c in decode(ap,root,cache) if c['name'].replace('\\','/') in selected)
    return result

def weapon_actions(data):
    """Resolve AIS1 animation references before considering legacy inline paths."""
    root=parse_i3r2(data);by_id={b.block_id:b for b in root.blocks};actions={}
    for block in root.blocks:
        if block.type_name!='i3AIState':continue
        name,offset=short(block.data);payload=block.data[offset:];resource=None
        if payload[:4]==b'AIS1' and len(payload)>=20:
            index,tag=struct.unpack_from('<HH',payload,16)
            if tag==0xffff:
                # External references directly index the resource string table.
                if index<len(root.text_lines):resource=root.text_lines[index]
            else:
                linked=by_id.get(index)
                if linked is not None and linked.type_name=='i3Animation':
                    if tag&0x8000 and tag-0x8000==linked.target and 0<=linked.target<len(root.text_lines):
                        resource=root.text_lines[linked.target]
        if resource and resource.lower().endswith('.i3a'):
            actions[name.lower()]=resource.replace('\\','/')
        else:
            resources=re.findall(rb'[A-Za-z0-9_./\\-]+\.i3a',block.data,re.IGNORECASE)
            if resources:actions[name.lower()]=resources[-1].decode('cp1252')
    return actions


def weapon_companion(stem,clips,actions,gender,side=''):
    half=re.search(r'_(Left|Right)$',stem,re.I)
    if side and half:stem=stem[:half.start()] if half[1].lower()==side.lower() else 'AttackIdle'
    stem=re.sub(r'_\d+$','',stem)
    norm=normalized(stem);action={'attackidle':'idle','reload':'loadmag','reloadc':'loadbullet','attack':'fire','attackburst':'fire','attacksemi':'fire','attacka':'attacka','attackb':'attackb','attackstab':'secondaryfire'}.get(norm.removesuffix('dual'),norm)
    keys={normalized(k):v for k,v in actions.items()}
    suffixes=([side+'_Folded','_Folded'] if side else [])+[gender+'1PV'+side,gender+side+'1PV','1PV'+gender+side,'1PV'+side,gender+side,side]
    for a in dict.fromkeys([action,norm]):
        for suffix in suffixes:
            ref=keys.get(normalized(a+suffix))
            if ref:
                match=next((c for c in clips if normalized(c['name'])==normalized(ref)),None)
                if match:return match
    wanted=[normalized(stem+x) for x in suffixes]
    for key in wanted:
        match=next((c for c in clips if normalized(Path(c['name']).stem)==key),None)
        if match:return match
    if norm!='attackidle':return weapon_companion('AttackIdle',clips,actions,gender,side)
    return next((c for c in clips if normalized(Path(c['name']).stem)=='idle'),None)

def assemble_dual(scene):
    # Runtime dual classes instantiate the same authored weapon hierarchy twice.
    model=scene['weapon'];arms=scene['arms']
    if not arms or 'WeaponPointDummyLeft' not in arms['sockets']:return
    original=copy.deepcopy(model);count=len(model['bones'])
    aw=sample_world(arms,None);restOffset=np.linalg.inv(aw[arms['sockets']['WeaponPointDummyRight']])@aw[arms['sockets']['WeaponPointDummyLeft']]
    for b in original['bones']:
        b=copy.deepcopy(b);world=restOffset@matrix(b['world']);b['world']=trs(world);b['inverse']=np.linalg.inv(world).T.flatten().tolist()
        if b['parent']<0:b['local']=trs(restOffset@matrix(b['local']))
        b['name']='left__'+b['name'];b['parent']=b['parent']+count if b['parent']>=0 else -1;model['bones'].append(b)
    for m in original['meshes']:
        m=copy.deepcopy(m);m['positions']=(restOffset@np.c_[m['positions'],np.ones(len(m['positions']))].T).T[:,:3].tolist();m['normals']=(np.linalg.inv(restOffset[:3,:3]).T@np.array(m['normals']).T).T.tolist();m['name']='Left / '+m['name'];m['boneIndices']=[[i+count for i in row] for row in m['boneIndices']];model['meshes'].append(m)
    for c in scene['clips']:
        c['socket']='WeaponPointDummyRight'
        source=c.pop('leftWeapon',None) or c['weapon'];tracks=copy.deepcopy((c['weapon'] or {}).get('tracks',[]))
        left=copy.deepcopy((source or {}).get('tracks',[]))
        for t in left:t['name']='left__'+t['name']
        for i,b in enumerate(original['bones']):
            if b['parent']>=0:continue
            samples=[]
            for f in range(sample_count(c['duration'])):
                aw=sample_world(arms,c['arms'],f);ww=sample_world(original,source,f)
                relative=np.linalg.inv(aw[arms['sockets']['WeaponPointDummyRight']])@aw[arms['sockets']['WeaponPointDummyLeft']]@ww[i]
                samples.append(trs(relative))
            left=[t for t in left if t['name']!='left__'+b['name']];left.append(dict(name='left__'+b['name'],flags=7,samples=samples))
        c['weapon']=dict(name=c['source']+' / paired weapon instances',duration=c['duration'],tracks=tracks+left)
    scene['dualInstances']=True

def matrix(v):
    x,y,z,w=v[3:7];m=np.eye(4)
    m[:3,:3]=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])@np.diag(v[7:10]);m[:3,3]=v[:3];return m

def sample_world(model,clip,frame=0):
    locals=[list(b['local']) for b in model['bones']];ids={b['name']:i for i,b in enumerate(model['bones'])}
    for t in (clip or {}).get('tracks',[]):
        if t['name'] not in ids or not t['samples']:continue
        v=t['samples'][min(frame,len(t['samples'])-1)];dst=locals[ids[t['name']]]
        for a,b,mask in [(0,3,1),(3,7,2),(7,10,4)]:
            if t['flags']&mask:dst[a:b]=v[a:b]
    world=[]
    for bone,v in zip(model['bones'],locals):
        m=matrix(v);world.append(world[bone['parent']]@m if bone['parent']>=0 else m)
    return world

def world_sampler(model,clip,targets):
    needed=set()
    for index in targets:
        while index>=0 and index not in needed:needed.add(index);index=model['bones'][index]['parent']
    ids={b['name']:i for i,b in enumerate(model['bones'])};bindings={}
    for track in (clip or {}).get('tracks',[]):
        if track['name'] in ids:bindings.setdefault(ids[track['name']],[]).append(track)
    ordered=[(i,model['bones'][i],bindings.get(i,[])) for i in sorted(needed)]
    def evaluate(frame):
        world={}
        for i,bone,tracks in ordered:
            value=list(bone['local'])
            for track in tracks:
                if not track['samples']:continue
                sample=track['samples'][min(frame,len(track['samples'])-1)]
                for a,b,flag in [(0,3,1),(3,7,2),(7,10,4)]:
                    if track['flags']&flag:value[a:b]=sample[a:b]
            local=matrix(value);parent=bone['parent'];world[i]=world[parent]@local if parent>=0 else local
        return world
    return evaluate

def resolve_attachments(scene):
    arms=scene['arms'];weapon=scene['weapon']
    if not arms:return
    sockets=arms['sockets'];right='WeaponPointDummyRight';left='WeaponPointDummyLeft'
    if right not in sockets:raise ValueError('Character has no right weapon socket')
    idle=next((c for c in scene['clips'] if c['name'] in ('AttackIdle','AttackIdle_Right','AttackIdle_Left')),None)
    if idle is None:return
    aw=sample_world(arms,idle['arms']);ww=sample_world(weapon,idle['weapon'])
    # Compare the same rigid body in the paired entry poses. This resolves the
    # client's per-weapon left/right reload flag from authored transforms, rather
    # than assuming every reload uses one hand or maintaining weapon-name offsets.
    main=max(weapon['meshes'],key=lambda m:len(m['positions']))
    owner=main['boneIndices'][0][0];reference=aw[sockets[right]]@ww[owner]
    for c in scene['clips']:
        c['socket']=left if c['name'].lower().endswith('_left') else right;c['attachmentEvidence']='source character socket'
        if not c['arms']:continue
        cw=sample_world(weapon,c['weapon']);ca=sample_world(arms,c['arms'])
        is_reload=c['name'].lower().startswith('reload')
        if is_reload and left in sockets:
            scores={}
            for name in [right,left]:
                m=ca[sockets[name]]@cw[owner]
                scores[name]=float(np.linalg.norm(m[:3,3]-reference[:3,3])+0.15*np.linalg.norm(m[:3,:3]-reference[:3,:3]))
            c['attachmentScores']=scores
            c['socket']=min(scores,key=scores.get)
            c['attachmentEvidence']='paired animation entry-pose inference; source sockets and source weapon tracks'
        c['attachmentBone']=sockets[c['socket']]
    paired=[c for c in scene['clips'] if c['arms']]
    for c in scene['clips']:
        if c['arms'] or not c['weapon']:continue
        matches=[x for x in paired if x['weapon'] and x['weapon']['name']==c['weapon']['name']]
        if matches:
            match=next((x for x in matches if x['name']=='AttackIdle'),matches[0])
            c['arms']=match['arms'];c['socket']=match['socket'];c['attachmentEvidence']=match['attachmentEvidence']

def flatten(x):return [v for row in x for v in row]
def write_model(model,path,cache):
    cast=Cast();root=cast.CreateRoot();meta=root.CreateMetadata();meta.SetSoftware('pb2cast');meta.SetUpAxis('z');m=root.CreateModel();m.SetName(path.stem);sk=m.CreateSkeleton()
    for b in model['bones']:
        bone=sk.CreateBone();bone.SetName(b['name']);bone.SetParentIndex(b['parent']);bone.SetLocalPosition(b['local'][:3]);bone.SetLocalRotation(b['local'][3:7]);bone.SetScale(b['local'][7:]);bone.SetWorldPosition(b['world'][:3]);bone.SetWorldRotation(b['world'][3:7])
    mats={};copied=set()
    for key,slots in model['materials'].items():
        mat=m.CreateMaterial();mat.SetName(key);mat.SetType('pbr');mats[key]=mat.hash
        for slot,name in slots.items():
            dest=path.parent/'textures'/name;dest.parent.mkdir(exist_ok=True)
            if name not in copied and (cache/name).resolve()!=dest.resolve():shutil.copy2(cache/name,dest)
            copied.add(name)
            f=mat.CreateFile();f.SetPath('textures/'+name);mat.SetSlot(slot,f.hash)
    for data in model['meshes']:
        mesh=m.CreateMesh();mesh.SetName(data['name']);mesh.SetFaceBuffer(data['indices']);mesh.SetVertexPositionBuffer(data['positions']);mesh.SetVertexNormalBuffer(data['normals']);mesh.SetUVLayerCount(1);mesh.SetVertexUVLayerBuffer(0,data['uvs']);mesh.SetMaximumWeightInfluence(len(data['weights'][0]));mesh.SetVertexWeightBoneBuffer(flatten(data['boneIndices']));mesh.SetVertexWeightValueBuffer(flatten(data['weights']));mesh.SetMaterial(mats[data['material']])
    cast.save(str(path));Cast.load(str(path))
def write_animation(clip,model,path):
    cast=Cast();root=cast.CreateRoot();a=root.CreateAnimation();a.SetName(path.stem);a.SetFramerate(30);a.SetLooping('idle' in clip['name'].lower())
    boneNames={b['name'] for b in model['bones']}; count=0
    for track in clip['tracks']:
        # Preserve all source tracks, including nodes absent from an individual model.
        values=track['samples'];frames=list(range(len(values)));flags=track['flags']
        for prop,indices,mask in [('tx',[0],1),('ty',[1],1),('tz',[2],1),('rq',[3,4,5,6],2),('sx',[7],4),('sy',[8],4),('sz',[9],4)]:
            if not flags&mask:continue
            curve=a.CreateCurve();curve.SetNodeName(track['name']);curve.SetKeyPropertyName(prop);curve.SetKeyFrameBuffer(frames);curve.SetMode('absolute')
            if prop=='rq':curve.SetVec4KeyValueBuffer([row[3:7] for row in values])
            else:curve.SetFloatKeyValueBuffer([row[indices[0]] for row in values])
            count+=1
    if not count:raise ValueError('Clip has no tracks matching the exported skeleton')
    cast.save(str(path));Cast.load(str(path))
def full_animation(scene,c, export_names=True):
    # Keep the two rigs independent while baking; shared source names are legal.
    armtracks=list((c.get('arms') or {}).get('tracks',[]))
    tracks={t['name']:dict(t) for t in (c.get('weapon') or {}).get('tracks',[])}
    # Work in source bone names while baking placement. Export names are applied
    # afterward to both weapon model bones and weapon animation tracks.
    if c.get('arms') and c.get('socket'):
        socket=scene['arms']['sockets'][c['socket']]
        roots=[(i,b) for i,b in enumerate(scene['weapon']['bones']) if b['parent']<0]
        armWorld=world_sampler(scene['arms'],c['arms'],[socket]);weaponWorld=world_sampler(scene['weapon'],c['weapon'],[i for i,b in roots])
        baked={i:[] for i,b in roots}
        for f in range(sample_count(c['duration'])):
            a=armWorld(f)[socket];worlds=weaponWorld(f)
            for i,b in roots:baked[i].append(trs(a@worlds[i]))
        for i,b in roots:tracks[b['name']]=dict(name=b['name'],flags=7,samples=baked[i])
    weapontracks=list(tracks.values())
    if export_names:
        for t in weapontracks:t['name']=weapon_export_name(t['name'])
    return dict(name=c['name'],duration=c['duration'],tracks=armtracks+weapontracks)

def weapon_export_name(name):return 'pb2cast_weapon__'+name

def export_weapon_model(model):
    model={**model,'bones':[{**b,'name':weapon_export_name(b['name'])} for b in model['bones']]}
    return model

CATEGORY_PREFIX={'Assault rifles':'AR','SMGs':'SMG','Sniper rifles':'sniper','Pistols':'pistol','Knives / melee':'knife','Shotguns':'shotgun','Machine guns':'MG','Grenades':'grenade','Launchers':'launcher','Bows':'bow','Equipment':'equipment'}
def export_stem(scene,key):
    name=scene.get('assetName',scene[key]['name']) if key=='weapon' else scene[key]['name']
    if key=='weapon':
        root=Path(scene['source']).parent.parent
        category=scene.get('category') or next((g['category'] for g in family_catalog(root) if Path(scene['source']).name in g['variants']),'Other / unclassified')
        return 'viewmodel_'+CATEGORY_PREFIX.get(category,'other')+'_'+clean(name)
    name=re.sub(r'_(viewarms|viewhands|playermodel)$','',name)
    return ('playermode_'+clean(name)+'_fb') if key=='player' else ('viewmodel_'+clean(name)+'_hands')

def animation_families(root):
    return sorted(p.name for p in (Path(root)/'Chara').iterdir() if p.is_dir() and not p.name.lower().startswith('dino') and any(p.glob('*.i3AnimPack')))

def player_animation_catalog(req):
    scene=json.loads(Path(req['scene']).read_text());root=Path(scene['source']).parent.parent;family=req['family']
    if family not in animation_families(root):raise ValueError('Choose an installed character animation family')
    entries=[];seen=set()
    for p in sorted((root/'Chara'/family).glob('*.i3AnimPack')):
        d=p.read_bytes()
        if d[:4] not in (b'APF1',b'APF2',b'APF3'):continue
        for i in range(u(d,4)):
            name=span(d,184+i*284,260).split(b'\0')[0].decode('cp1252');name=name.replace('\\','/')
            if '/3pv/' not in name.lower() or name.lower() in seen:continue
            stem=Path(name).stem.lower()
            if stem.startswith(('low_','down_')):continue
            kind='torso' if stem.startswith('up_') else 'body'
            entries.append(dict(name=name.split('/3pv/')[-1],source=name,pack=str(p),kind=kind,family=family));seen.add(name.lower())
    scene['playerAnimations']=entries;save_json(req['scene'],scene)
    return dict(ok=True,message=f'{len(entries)} player animations indexed.',playerAnimations=entries)

def export_player_animations(scene,req,cache):
    model=scene.get('player')
    if not model:raise ValueError('Load a corresponding player model before exporting its animations')
    allowed={b['name'] for b in model['bones']};upper=set()
    for b in model['bones']:
        if b['name'].lower().startswith('spine') or (b['parent']>=0 and model['bones'][b['parent']]['name'] in upper):upper.add(b['name'])
    groups={}
    for i in req.get('playerClips',[]):
        e=scene['playerAnimations'][i];groups.setdefault(e['pack'],[]).append(e)
    dest=Path(req['animationDestination']);dest.mkdir(parents=True,exist_ok=True);written=[]
    for packpath,entries in groups.items():
        clips={c['name'].replace('\\','/').lower():c for c in decode(Path(packpath),Path(scene['source']).parent.parent,cache)}
        for e in entries:
            original=clips[e['source'].lower()];valid=upper if e['kind']=='torso' else allowed
            c={**original,'tracks':[t for t in original['tracks'] if t['name'] in valid]}
            if not c['tracks']:raise ValueError('No matching player bones for '+e['source'])
            prefix='pt_' if e['kind']=='torso' else 'pb_'
            name=prefix+clean(model['name'].removesuffix('_playermodel'))+'_'+clean(e['name'].removesuffix('.i3a').replace('/','_'))
            path=dest/(name+'.cast');write_animation(c,model,path);written.append(str(path))
    return written

def export_all_characters(req):
    root=Path(req['root']);cache=Path(req['cache']);cache.mkdir(parents=True,exist_ok=True)
    dest=Path(req['modelDestination']);dest.mkdir(parents=True,exist_ok=True)
    entries=[];results=[];files=[];counts={'player':0,'arms':0};used=set()
    for package in sorted((root/'Pack').glob('Chara*.i3Pack')):
        try:
            for name,data in package_assets(package).items():
                if not name.endswith('.i3s'):continue
                parsed=blocks(data)
                palettes=[b for k,b in parsed.values() if k=='i3BoneMatrixListAttr']
                human=any({'R Hand','L Hand'} <= {b[48+i*128:80+i*128].split(b'\0')[0].decode('cp1252') for i in range(u(b,4))} for b in palettes)
                if human:entries.append((package,name))
        except Exception as e:results.append(dict(package=package.name,status='scan failed',error=str(e)))
    save_json(cache/'progress.json',dict(done=0,total=len(entries),message='Character models indexed'))
    for index,(package,name) in enumerate(entries):
        assets=package_assets(package);data=assets[name];stem=package.stem.removeprefix('Chara_')
        if resource_identity(Path(name).stem)!=resource_identity(stem):stem+='_'+Path(name).stem
        modelcache=cache/clean(package.stem)/clean(Path(name).stem)
        for kind,suffix in [('player','_playermodel'),('arms','_viewhands')]:
            entry=dict(package=package.name,model=name,kind=kind)
            save_json(cache/'progress.json',dict(done=index,total=len(entries),message='Exporting '+stem+' '+('player model' if kind=='player' else 'viewhands')))
            try:
                model=read_character(data,stem+suffix,assets,root,modelcache,arms=kind=='arms',player=kind=='player')
                filename=('playermode_'+clean(stem)+'_fb' if kind=='player' else 'viewmodel_'+clean(stem)+'_hands')+'.cast'
                if filename.lower() in used:filename=Path(filename).stem+'_'+hashlib.sha1((package.name+'/'+name).encode()).hexdigest()[:8]+'.cast'
                used.add(filename.lower());output=dest/filename;write_model(model,output,modelcache)
                counts[kind]+=1;files.append(str(output));entry.update(status='exported',file=str(output))
            except Exception as e:entry.update(status='failed',error=str(e))
            results.append(entry)
        save_json(cache/'progress.json',dict(done=index+1,total=len(entries),message=f"Exported {counts['player']} players and {counts['arms']} viewhands"))
    report=dest/'pb2cast_character_export_report.json';save_json(report,dict(counts=counts,entries=results))
    failures=[x for x in results if x['status']!='exported']
    details='Report: '+str(report)+'\n'+'\n'.join(x['package']+' / '+x.get('model','')+' / '+x.get('kind','scan')+': '+x['error'] for x in failures)
    return dict(ok=bool(files),message=f"Exported {counts['player']} player models and {counts['arms']} viewhands separately; {len(failures)} failures.",files=files,counts=counts,report=str(report),details=details)

def batch_export(req):
    files=[];errors={};cache=Path(req['cache']);cache.mkdir(parents=True,exist_ok=True)
    for p in req['packs']:
        try:
            scenePath=req.get('preloaded',{}).get(p)
            if not scenePath or not Path(scenePath).exists():scenePath=load(dict(root=req['root'],pack=p,arms=req.get('armGender','Male'),cache=str(cache/clean(p))))['scene']
            scene=json.loads(Path(scenePath).read_text());child={**req,'scene':scenePath,'clips':list(range(len(scene['clips']))),'parts':[],'playerClips':[]}
            files+=export(child,scene)['files']
        except Exception as e:errors[p]=str(e)
    return dict(ok=True,message=f'Batch exported {len(files)} files; {len(errors)} weapons failed.',files=list(dict.fromkeys(files)),details='\n'.join(k+': '+v for k,v in errors.items()))

def export(req,scene=None):
    if scene is None:scene=json.loads(Path(req['scene']).read_text())
    if scene.get('samplingRevision') != DECODER_REV or scene.get('pipelineRevision') != PIPELINE_REV:
        raise ValueError('This scene uses an older exporter. Click Load assets before exporting again.')
    cache=Path(scene['cache']);written=[];weaponStem=export_stem(scene,'weapon') if scene.get('weapon') else None
    if req.get('models',True):
        dest=Path(req['modelDestination']);dest.mkdir(parents=True,exist_ok=True)
        for key in ('weapon','arms','player'):
            model=scene.get(key)
            if model and req.get(key,key!='player'):
                selected=req.get('parts',[])
                if key=='weapon' and selected:model={**model,'meshes':[m for m in model['meshes'] if m['name'] in selected]}
                if not model['meshes']:raise ValueError('Select at least one mesh')
                path=dest/((weaponStem if key=='weapon' else export_stem(scene,key))+'.cast');write_model(export_weapon_model(model) if key=='weapon' else model,path,cache);written.append(str(path))
    if req.get('animations',True):
        dest=Path(req['animationDestination']);dest.mkdir(parents=True,exist_ok=True)
        for i in dict.fromkeys(req.get('clips',[])):
            c=scene['clips'][i];clip=full_animation(scene,c,export_names=True)
            path=dest/(weaponStem+'_'+clean(c['name'])+'.cast')
            write_animation(clip,scene['weapon'],path);written.append(str(path))
    if req.get('playerClips'):written+=export_player_animations(scene,req,cache)
    if not written:raise ValueError('Nothing selected for export')
    return dict(ok=True,message=f'Exported and read back {len(written)} CAST files.',files=written)
def main():
    req=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));out=Path(req['response'])
    try:result=export_all_characters(req) if req['action']=='all_character_models' else gallery(req) if req['action']=='gallery' else dict(ok=True,message='Weapon families indexed.',families=family_catalog(req['root'])+viewhands_catalog(req['root']),animationFamilies=animation_families(req['root'])) if req['action']=='catalog' else player_animation_catalog(req) if req['action']=='player_catalog' else batch_export(req) if req['action']=='batch_export' else preload_category(req) if req['action']=='preload' else load(req) if req['action']=='load' else export(req)
    except Exception as ex:result=dict(ok=False,message=str(ex),details=traceback.format_exc())
    save_json(out,result)
    return 0 if result['ok'] else 1
if __name__=='__main__':sys.exit(main())

