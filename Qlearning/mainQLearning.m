%%Main loop for reinforcement learning.

hold off
clf;

k = 3;
gamma = 0.7;
alpha = 0.5;

gwinit(k);
s = gwstate;
x = s.xsize;
y = s.ysize;

Q = rand(x, y, 4);

Q(:,1,4) = -inf;
Q(1,:,2) = -inf;
Q(:,y,3) = -inf;
Q(x,:,1) = -inf;
max_iter = 100;

for i = 1:max_iter
    if i <= max_iter%/5
        epsilon = 1;
    else
        epsilon = 1-(i-max_iter/5)/(4/5*max_iter);
    end
    gwinit(k);
    while 1
    s = gwstate;
    stat = sample([0,1], [epsilon, 1-epsilon]);
    if stat
        [M, action] = max(Q(s.pos(1),s.pos(2),:));
    else
        action = sample([1:4], 0.25*ones(1,4));
    end
    
    prevPos = s.pos;
    state = gwaction(action);
    %gwdraw;
    if (not(state.isvalid))
        %pos = state.pos;
        %Q(pos(1), pos(2), action) = -inf;
        continue;
    end

    if (state.isterminal)
        Q(prevPos(1), prevPos(2), action) = (1-alpha)*Q(prevPos(1), prevPos(2), action) + alpha*state.feedback;
        break;
    else
        Q(prevPos(1), prevPos(2), action) = (1-alpha)*Q(prevPos(1), prevPos(2), action) + alpha*(state.feedback + gamma*max(Q(state.pos(1), state.pos(2), :)));
    end
    end
    if (mod(i,100)==0)
       i
    end
end
figure(1)
gwdraw;
hold on
gwplotallarrows(Q)
hold off

figure(2)
for i=1:4
subplot(2,2,i)
imagesc(Q(:,:,i));
end


%gwdraw;