import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import Plotly from "plotly.js-dist-min";
import "../assets/style.css";

const $ = (id) => document.getElementById(id);
const colors = {trajectory:"#8b95a5", pushback:"#ff9f1c", push:"#1677ff", pull:"#9c36b5", active:"#e31a1c", preview:"#facc15"};
let state = {loaded:false, ids:[], flightId:null, flight:null, annotation:null, active:0, start:null, end:null, tags:[], counts:{}, statuses:{}};

const emptyFC = () => ({type:"FeatureCollection",features:[]});
const api = async (url, options={}) => { const r=await fetch(url,{headers:{"Content-Type":"application/json"},...options}); if(!r.ok) throw new Error((await r.json()).detail||r.statusText); return r.json(); };
const feature = (idx,color,radius=4) => ({type:"Feature",id:idx,geometry:{type:"Point",coordinates:[state.flight.longitude[idx],state.flight.latitude[idx]]},properties:{idx,color,radius}});
const validPoint = i => Number.isFinite(state.flight.latitude[i]) && Number.isFinite(state.flight.longitude[i]);

const basemaps={osm:{label:"OpenStreetMap",attribution:"© OpenStreetMap contributors"},"carto-light":{label:"CARTO Light",attribution:"© OpenStreetMap contributors © CARTO"},"carto-dark":{label:"CARTO Dark",attribution:"© OpenStreetMap contributors © CARTO"}};
const baseSources=Object.fromEntries(Object.entries(basemaps).map(([id,x])=>[id,{type:"raster",tiles:[`${location.origin}/tiles/${id}/{z}/{x}/{y}.png`],tileSize:256,maxzoom:19,attribution:x.attribution}]));
const baseLayers=Object.keys(basemaps).map(id=>({id:`base-${id}`,type:"raster",source:id,layout:{visibility:id==="osm"?"visible":"none"},paint:{"raster-fade-duration":0}}));
const map = new maplibregl.Map({container:"map",style:{version:8,sources:baseSources,layers:baseLayers},center:[0,0],zoom:16,maxZoom:22,maxTileCacheSize:256});
map.addControl(new maplibregl.NavigationControl(),"top-right");
map.on("load",()=>{
  for(const id of ["trajectory","preview","active"]) map.addSource(id,{type:"geojson",data:emptyFC()});
  map.addLayer({id:"trajectory",type:"circle",source:"trajectory",paint:{"circle-radius":["get","radius"],"circle-color":["get","color"],"circle-stroke-width":1,"circle-stroke-color":["get","color"]}});
  map.addLayer({id:"preview",type:"circle",source:"preview",paint:{"circle-radius":["get","radius"],"circle-color":colors.preview,"circle-opacity":.9}});
  map.addLayer({id:"active",type:"circle",source:"active",paint:{"circle-radius":9,"circle-color":colors.active,"circle-stroke-width":2,"circle-stroke-color":"#fff"}});
  map.on("click","trajectory",e=>{ if(e.features?.length) setActive(Number(e.features[0].properties.idx)); });
  map.on("mouseenter","trajectory",()=>map.getCanvas().style.cursor="pointer"); map.on("mouseleave","trajectory",()=>map.getCanvas().style.cursor="");
  if(state.flight){updateMapLayers();focusMap();}
});
function setBasemap(id){for(const key of Object.keys(basemaps))if(map.getLayer(`base-${key}`))map.setLayoutProperty(`base-${key}`,"visibility",key===id?"visible":"none");}

function pointColors(){
  const out=Array(state.flight.timestamp.length).fill(colors.trajectory);
  for(const kind of ["pushback","push","pull"]) for(const s of state.annotation.segments[kind]||[]){const lo=Math.min(s.start_idx,s.end_idx),hi=Math.max(s.start_idx,s.end_idx);for(let i=lo;i<=hi&&i<out.length;i++)out[i]=colors[kind];}
  return out;
}
function mapFeatures(){const c=pointColors();return {type:"FeatureCollection",features:c.map((x,i)=>validPoint(i)?feature(i,x,Math.max(3,Number($("point-size").value)/2)):null).filter(Boolean)};}
function updateMapLayers(){
  if(!map.isStyleLoaded()||!state.flight)return;
  map.getSource("trajectory").setData(mapFeatures());
  map.getSource("active").setData({type:"FeatureCollection",features:validPoint(state.active)?[feature(state.active,colors.active,9)]:[]});
  const p=[]; if(state.start!==null&&state.end!==null){const lo=Math.min(state.start,state.end),hi=Math.max(state.start,state.end);for(let i=lo;i<=hi;i++)if(validPoint(i))p.push(feature(i,colors.preview,Math.max(5,Number($("point-size").value)/2+2)));}
  map.getSource("preview").setData({type:"FeatureCollection",features:p});
}
function focusMap(){
  if(!state.flight)return; const n=Math.min(state.flight.latitude.length,Math.max(2,Number($("focus-points").value)||180)); const coords=[];for(let i=0;i<n;i++)if(validPoint(i))coords.push([state.flight.longitude[i],state.flight.latitude[i]]);if(!coords.length)return;
  const lon=coords.reduce((a,x)=>a+x[0],0)/coords.length,lat=coords.reduce((a,x)=>a+x[1],0)/coords.length;map.jumpTo({center:[lon,lat],zoom:Number($("zoom").value)||16});
}
function createPlot(){
  const f=state.flight,c=pointColors(); const focus=Math.min(f.timestamp.length,Math.max(2,Number($("plot-focus-points").value)||180));
  Plotly.newPlot("gs",[
    {x:f.timestamp,y:f.gs,mode:"lines+markers",line:{color:"#7b8799",width:1},marker:{size:6,color:c},customdata:f.timestamp.map((_,i)=>i),hovertemplate:"Punkt %{customdata}<br>%{x}<br>GS %{y:.2f}<extra></extra>"},
    {x:[f.timestamp[0]],y:[f.gs[0]],mode:"markers",marker:{size:14,color:colors.active},hoverinfo:"skip"},
    {x:[],y:[],mode:"markers",marker:{size:9,color:colors.preview},hoverinfo:"skip"}
  ],{height:390,margin:{l:55,r:15,t:8,b:45},dragmode:"pan",showlegend:false,xaxis:{title:"Zeit",range:[f.timestamp[0],f.timestamp[focus-1]],rangeslider:{visible:true,thickness:.12}},yaxis:{title:"compute_gs [kt]",range:[0,Number($("max-gs").value)||20]},uirevision:`gs-${state.flightId}`},{scrollZoom:true,displaylogo:false,responsive:true});
  $("gs").on("plotly_click",e=>{if(e.points?.length)setActive(Number(e.points[0].customdata));});
}
function focusPlot(){if(!state.flight)return;const n=Math.min(state.flight.timestamp.length,Math.max(2,Number($("plot-focus-points").value)||180));Plotly.relayout("gs",{"xaxis.range":[state.flight.timestamp[0],state.flight.timestamp[n-1]]});}
function updatePlot(){
  if(!state.flight)return; const f=state.flight,c=pointColors(); Plotly.restyle("gs",{"marker.color":[c]},[0]); Plotly.restyle("gs",{x:[[f.timestamp[state.active]]],y:[[f.gs[state.active]]]},[1]);
  if(state.start===null||state.end===null)Plotly.restyle("gs",{x:[[]],y:[[]]},[2]);else{const lo=Math.min(state.start,state.end),hi=Math.max(state.start,state.end);Plotly.restyle("gs",{x:[f.timestamp.slice(lo,hi+1)],y:[f.gs.slice(lo,hi+1)]},[2]);}
}
function updateActivePoint(){
  if(!state.flight)return;
  if(map.isStyleLoaded()&&map.getSource("active"))map.getSource("active").setData({type:"FeatureCollection",features:validPoint(state.active)?[feature(state.active,colors.active,9)]:[]});
  Plotly.restyle("gs",{x:[[state.flight.timestamp[state.active]]],y:[[state.flight.gs[state.active]]]},[1]);
}
function setActive(idx){state.active=Math.max(0,Math.min(idx,state.flight.timestamp.length-1));updateActivePoint();renderPointInfo();}
function renderPointInfo(){const i=state.active,dt=i?((new Date(state.flight.timestamp[i])-new Date(state.flight.timestamp[i-1]))/1000):0;$("point-info").textContent=`Aktiver Punkt: ${i} / ${state.flight.timestamp.length-1} · ${state.flight.timestamp[i]} · Δt ${dt.toFixed(1)} s · GS ${Number(state.flight.gs[i]??0).toFixed(2)} kt`;const duration=state.start!==null&&state.end!==null?Math.abs((new Date(state.flight.timestamp[state.end])-new Date(state.flight.timestamp[state.start]))/1000):null;$("selection-info").textContent=`Start: ${state.start??"–"} | Ende: ${state.end??"–"}${duration===null?"":` | Dauer: ${duration.toFixed(1)} s`}`;}

function renderSegments(){const box=$("segments");box.innerHTML="";for(const kind of ["pushback","push","pull"])state.annotation.segments[kind].forEach((s,i)=>{const row=document.createElement("div");row.className="segment-row";row.innerHTML=`<strong style="color:${colors[kind]}">${kind}</strong><span>Punkt ${s.start_idx}</span><span>Punkt ${s.end_idx}</span><button class="delete-x">×</button>`;row.querySelector("button").onclick=()=>{state.annotation.segments[kind].splice(i,1);if(kind==="pushback"&&!state.annotation.segments.pushback.length&&state.annotation.status==="labelled")state.annotation.status="not_labelled";saveAnnotation();renderAll();};box.append(row);});if(!box.children.length)box.innerHTML='<span class="small-label">Noch keine Segmente</span>';}
function renderToggles(){
  const sb=$("status-buttons");sb.innerHTML="";if(state.annotation.status==="labelled"){const x=document.createElement("span");x.className="toggle status-labelled active";x.textContent="✓ Gelabelt (automatisch)";sb.append(x);}for(const [key,label] of [["not_labelled","Not done yet"],["no_pushback","No pushback"],["not_clear","Not clear"]]){const b=document.createElement("button");b.className=`toggle ${state.annotation.status===key?"active":""}`;b.textContent=label;b.onclick=()=>{state.annotation.status=key;renderToggles();};sb.append(b);}
  const tb=$("tag-buttons");tb.innerHTML="";for(const tag of state.tags){const b=document.createElement("button");b.className=`toggle ${state.annotation.tags.includes(tag)?"active":""}`;b.textContent=tag;b.onclick=()=>{const i=state.annotation.tags.indexOf(tag);i<0?state.annotation.tags.push(tag):state.annotation.tags.splice(i,1);renderToggles();};tb.append(b);}
}
function renderMetrics(){const c=state.counts,total=state.ids.length;$("m-total").textContent=total;$("m-labelled").textContent=c.labelled||0;$("m-no").textContent=c.no_pushback||0;$("m-unclear").textContent=c.not_clear||0;$("m-open").textContent=c.not_labelled||0;$("progress-fill").style.width=`${100*(total-(c.not_labelled||0))/Math.max(1,total)}%`;}
function renderAll(){updateMapLayers();updatePlot();renderPointInfo();renderSegments();renderToggles();}
async function refreshStatus(){const s=await api("/api/status");state.ids=s.flights;state.tags=s.tags;state.counts=s.counts;state.statuses=s.statuses||{};for(const [key,id] of [["airport","meta-airport"],["date","meta-date"]])if(s.dataset?.[key])$(id).value=s.dataset[key];const info=document.querySelector(".derived-meta");if(s.dataset?.season)info.textContent=`Automatisch bestimmt: ${s.dataset.season} (Nordhalbkugel-Kalender)`;renderMetrics();const sel=$("flight");const current=state.flightId;sel.innerHTML=s.flights.map(x=>`<option value="${x}">${x}</option>`).join("");if(current)sel.value=current;}
async function loadFlight(id){state.flightId=id;state.flight=await api(`/api/flight/${encodeURIComponent(id)}`);state.annotation=state.flight.annotation;state.active=0;state.start=null;state.end=null;$("note").value=state.annotation.note||"";createPlot();renderAll();focusMap();$("flight").value=id;}
async function saveAnnotation(){state.annotation.note=$("note").value;await api(`/api/annotation/${encodeURIComponent(state.flightId)}`,{method:"PUT",body:JSON.stringify(state.annotation)});await refreshStatus();}
function addSegment(kind){if(state.start===null||state.end===null)return;const lo=Math.min(state.start,state.end),hi=Math.max(state.start,state.end);state.annotation.segments[kind].push({start_idx:lo,end_idx:hi,start_ts:state.flight.timestamp[lo],end_ts:state.flight.timestamp[hi]});state.annotation.segments[kind].sort((a,b)=>a.start_idx-b.start_idx);if(kind==="pushback")state.annotation.status="labelled";saveAnnotation();renderAll();}

$("load").onclick=async()=>{try{const columns={flight_id:$("col-flight").value,timestamp:$("col-time").value,latitude:$("col-lat").value,longitude:$("col-lon").value,groundspeed:$("col-gs").value,airport:$("col-airport").value,season:$("col-season").value};const metadata={airport:$("meta-airport").value,date:$("meta-date").value};const s=await api("/api/load",{method:"POST",body:JSON.stringify({source:$("data-path").value,output:$("output-path").value,columns,metadata})});state.loaded=true;state.ids=s.flights;state.tags=s.tags;state.counts=s.counts;renderMetrics();await refreshStatus();if(state.ids.length)await loadFlight(state.ids[0]);}catch(e){$("message").textContent=e.message;}};
$("flight").onchange=e=>loadFlight(e.target.value);$("prev-flight").onclick=()=>{const i=state.ids.indexOf(state.flightId);if(i>0)loadFlight(state.ids[i-1]);};$("next-flight").onclick=()=>{const i=state.ids.indexOf(state.flightId);if(i<state.ids.length-1)loadFlight(state.ids[i+1]);};
for(const [id,step] of [["minus10",-10],["minus1",-1],["plus1",1],["plus10",10]])$(id).onclick=()=>setActive(state.active+step);
$("set-start").onclick=()=>{state.start=state.active;renderAll();};$("set-end").onclick=()=>{state.end=state.active;renderAll();};$("clear-start").onclick=()=>{state.start=null;renderAll();};$("clear-end").onclick=()=>{state.end=null;renderAll();};
$("add-pushback").onclick=()=>addSegment("pushback");$("add-push").onclick=()=>addSegment("push");$("add-pull").onclick=()=>addSegment("pull");$("apply-map-view").onclick=focusMap;
$("apply-plot-view").onclick=focusPlot;
$("map-style").onchange=e=>setBasemap(e.target.value);
$("save").onclick=saveAnnotation;$("save-next").onclick=async()=>{await saveAnnotation();const open=state.ids.find(x=>x!==state.flightId&&state.statuses[x]==="not_labelled");if(open)loadFlight(open);};
$("add-tag").onclick=async()=>{const tag=$("new-tag").value.trim();if(!tag)return;const r=await api("/api/tags",{method:"PUT",body:JSON.stringify({tag})});state.tags=r.tags;$("new-tag").value="";renderToggles();};
