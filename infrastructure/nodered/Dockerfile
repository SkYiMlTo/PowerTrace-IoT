FROM nodered/node-red:latest
RUN npm install node-red-contrib-postgresql
COPY data/flows.json /data/flows.json
COPY data/flows_cred.json /data/flows_cred.json
COPY data/settings.js /data/settings.js
